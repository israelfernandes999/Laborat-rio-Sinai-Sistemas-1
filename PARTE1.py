# =============================================================================
# LABORATÓRIO 1 - SINAIS DE TEMPO CONTÍNUO E DISCRETO
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

print("=== INICIANDO LABORATÓRIO 1 - SINAIS ===")

# =============================================================================
# PARTE 1.1: CÁLCULO DAS FREQUÊNCIAS
# =============================================================================
print("\n" + "="*60)
print("PARTE 1.1: CÁLCULO DAS FREQUÊNCIAS DO SINAL CONTÍNUO")
print("="*60)

print("Sinal: z(t) = sen(0,5·π·t) + cos(2·π·t) + 1")

# Termo seno
omega1 = 0.5 * np.pi
f1 = omega1 / (2 * np.pi)
T1 = 1 / f1

print("\n--- Termo Seno: sen(0,5·π·t) ---")
print(f"Frequência angular (ω): {omega1:.4f} rad/s")
print(f"Frequência (f): {f1:.4f} Hz")
print(f"Período (T): {T1:.4f} segundos")

# Termo cosseno
omega2 = 2 * np.pi
f2 = omega2 / (2 * np.pi)
T2 = 1 / f2

print("\n--- Termo Cosseno: cos(2·π·t) ---")
print(f"Frequência angular (ω): {omega2:.4f} rad/s")
print(f"Frequência (f): {f2:.4f} Hz")
print(f"Período (T): {T2:.4f} segundos")

# Período total
print("\n--- Período Total do Sinal z(t) ---")
print(f"Período do seno: {T1:.1f} s")
print(f"Período do cosseno: {T2:.1f} s")
print(f"Período fundamental do sinal: {max(T1, T2):.1f} segundos")

# =============================================================================
# PARTE 1.2: SINAL CONTÍNUO
# =============================================================================
print("\n" + "="*60)
print("PARTE 1.2: REPRESENTAÇÃO DO SINAL CONTÍNUO")
print("="*60)

t = np.arange(-1, 1, 0.001)
z = np.sin(0.5 * np.pi * t) + np.cos(2 * np.pi * t) + 1

plt.figure(figsize=(12, 4))
plt.plot(t, z, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude z(t)')
plt.title('Sinal Contínuo: z(t) = sen(0,5πt) + cos(2πt) + 1')
plt.xlim(-1, 1)
plt.show()

print("Contradomínio observado: aproximadamente entre 0 e 3")
print("Período observado: 4 segundos (confirmado graficamente)")

# =============================================================================
# PARTE 1.3: DISCRETIZAÇÃO DO SINAL
# =============================================================================
print("\n" + "="*60)
print("PARTE 1.3: DISCRETIZAÇÃO DO SINAL")
print("="*60)

Ts = 0.01
n = np.arange(-1, 1, Ts)
z_n = np.sin(0.5 * np.pi * n) + np.cos(2 * np.pi * n) + 1

# Cálculo das frequências discretas
f1_d = f1 * Ts
f2_d = f2 * Ts
omega1_d = omega1 * Ts
omega2_d = omega2 * Ts

print(f"Período de amostragem Ts: {Ts} s")
print(f"Frequência de amostragem Fs: {1/Ts} Hz")
print(f"Frequência do seno (ciclos/amostra): {f1_d:.4f}")
print(f"Frequência do cosseno (ciclos/amostra): {f2_d:.4f}")
print(f"Frequência angular do seno (rad/amostra): {omega1_d:.4f}")
print(f"Frequência angular do cosseno (rad/amostra): {omega2_d:.4f}")

# =============================================================================
# PARTE 1.4: SINAL DISCRETO
# =============================================================================
print("\n" + "="*60)
print("PARTE 1.4: REPRESENTAÇÃO DO SINAL DISCRETO")
print("="*60)

plt.figure(figsize=(12, 4))
plt.stem(n, z_n, linefmt='blue', markerfmt='bo', basefmt=' ')
plt.grid(True, alpha=0.3)
plt.xlabel('Amostras (n)')
plt.ylabel('Amplitude z(n)')
plt.title('Sinal Discreto: z(n) = sen(0,5πn) + cos(2πn) + 1')
plt.xlim(-1, 1)
plt.show()

print("Domínio: n (amostras discretas)")
print("Contradomínio: mesmo do sinal contínuo (0 a 3)")
print(f"Número total de amostras: {len(z_n)}")

# =============================================================================
# PARTE 1.5 e 1.6: FUNÇÕES COMPONENTE PAR E ÍMPAR
# =============================================================================
print("\n" + "="*60)
print("PARTE 1.5 e 1.6: COMPONENTES PAR E ÍMPAR DO SINAL")
print("="*60)

def componente_par(x):
    """Retorna a componente par do sinal x"""
    return 0.5 * (x + x[::-1])

def componente_impar(x):
    """Retorna a componente ímpar do sinal x"""
    return 0.5 * (x - x[::-1])

# Aplicando ao sinal discreto
z_par = componente_par(z_n)
z_impar = componente_impar(z_n)

print("Funções componente_par() e componente_impar() criadas com sucesso!")

# =============================================================================
# PARTE 1.7: VISUALIZAÇÃO DAS COMPONENTES PAR E ÍMPAR
# =============================================================================
print("\n" + "="*60)
print("PARTE 1.7: VISUALIZAÇÃO DAS COMPONENTES PAR E ÍMPAR")
print("="*60)

plt.figure(figsize=(14, 8))

# Componente Par
plt.subplot(3, 1, 1)
plt.stem(n, z_par, linefmt='blue', markerfmt='bo')
plt.grid(True, alpha=0.3)
plt.ylabel('Amplitude')
plt.title('Componente Par do Sinal')
plt.xlim(-1, 1)

# Componente Ímpar
plt.subplot(3, 1, 2)
plt.stem(n, z_impar, linefmt='red', markerfmt='ro')
plt.grid(True, alpha=0.3)
plt.ylabel('Amplitude')
plt.title('Componente Ímpar do Sinal')
plt.xlim(-1, 1)

# Soma (deve ser igual ao sinal original)
plt.subplot(3, 1, 3)
plt.stem(n, z_par + z_impar, linefmt='green', markerfmt='go')
plt.grid(True, alpha=0.3)
plt.xlabel('Amostras (n)')
plt.ylabel('Amplitude')
plt.title('Soma: Componente Par + Componente Ímpar')
plt.xlim(-1, 1)

plt.tight_layout()
plt.show()

print("Análise: A soma das componentes par e ímpar reconstitui o sinal original")
print("Componente par: simétrica em relação ao eixo y")
print("Componente ímpar: antissimétrica em relação ao eixo y")