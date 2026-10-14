"""Transcrição de proj_P3DX_montagem.m — plotar robô e realizar posicionamento.

Exemplo de navegação no cenário da linha de montagem: o robô sai da Base e o
ponto de controle visita uma sequência de posições desejadas, trocadas por tempo.
Usa o controlador cinemático de posição do ponto de controle da AuRoRA
(cP3DX_KinematicController, incluído neste arquivo). Esse controlador leva o
PONTO DE CONTROLE até uma posição, mas não impõe a orientação final do robô.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from pioneer3dx import Pioneer3DX, tic, toc
from cenario_linha_montagem import cenario_linha_montagem, ESTACOES


# =========================================================================
def cP3DX_KinematicController(P, pgains=(0.35, 0.35, 0.8, 0.8)):
    """Equivalente Python de cP3DX_KinematicController (plataforma AuRoRA).

    Controlador cinemático de posição para o ponto de controle do Pioneer 3-DX:
        Ur = K^-1 * ( dXd + Kp1 * tanh(Kp2 * Xtil) )
    """
    Kp1 = np.diag(pgains[0:2])
    Kp2 = np.diag(pgains[2:4])

    psi = P.pPos.X[5]
    a = P.pPar.a
    K = np.array([[np.cos(psi), -a * np.sin(psi)],
                  [np.sin(psi),  a * np.cos(psi)]])

    P.pPos.Xtil = P.pPos.Xd - P.pPos.X

    P.pSC.Ur = np.linalg.solve(
        K, P.pPos.Xd[[6, 7]] + Kp1 @ np.tanh(Kp2 @ P.pPos.Xtil[[0, 1]]))
    P.pSC.Ud = P.pSC.Ur
    return P


# =========================================================================
# Limpar figuras
plt.close('all')

# =========================================================================
# Carregar a classe referente ao robô e colocá-lo na Base
A = Pioneer3DX()
xb, yb, angb = ESTACOES['Base']
A.rSetPose(xb, yb, np.deg2rad(angb))

# Posições desejadas do ponto de controle, trocadas a cada tmax/4 segundos.
# Todas ficam no corredor central, longe dos blocos.
DESTINOS = [(0.0, -2.5), (1.0, 0.0), (-1.0, 2.5), (0.0, -4.5)]

# =========================================================================
# Criar ambiente e exibir o robô
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
cenario_linha_montagem(ax)
ax.view_init(elev=30, azim=30 - 90)   # view(30,30)
# (lighting phong / material shiny / colormap winter não têm equivalente direto)

plt.ion()
plt.show()

A.mCADplot(ax)
A.mCADcolor(np.array([0, 51, 80]) / 255)
plt.pause(0.001)

XX = []  # Lista para armazenamento dos dados de navegação

Rastro_X = [A.pPos.X[0:3].copy()]   # Rastro atual
Rastro_Xd = [A.pPos.X[0:3].copy()]  # Rastro desejado

linhaX, = ax.plot([A.pPos.X[0]], [A.pPos.X[1]], [A.pPos.X[2]], '-k', zorder=10)
linhaXd, = ax.plot([A.pPos.X[0]], [A.pPos.X[1]], [A.pPos.X[2]], '--r', zorder=10)

plt.pause(1)
print('Início ............')

# =========================================================================
tmax = 100  # Tempo de simulação em segundos
Info = ax.set_title(f'Time: {0:05.2f} | {tmax:05.2f}s')

t = tic()   # Tempo corrente de simulação
tc = tic()  # Tempo de envio de sinal de controle
tp = tic()  # Tempo de plotagem do robô

while toc(t) < tmax:
    if toc(tc) > 1 / 10:   # 10 Hz
        tc = tic()

        # 0- Definir posição desejada (troca a cada tmax/4 segundos)
        k = min(int(toc(t) // (tmax / 4)), len(DESTINOS) - 1)
        A.pPos.Xd[[0, 1]] = DESTINOS[k]

        # Controlador
        # 1- Ler dados dos sensores
        A.rGetSensorData()

        # 2- Calcular sinais de controle
        A = cP3DX_KinematicController(A)

        # 3- Armazenar dados de navegação (LOG): [Xd(12) X(12) Ud(2) t]
        XX.append(np.r_[A.pPos.Xd, A.pPos.X, A.pSC.Ud, toc(t)])
        Rastro_X.append(A.pPos.X[0:3].copy())
        Rastro_Xd.append(A.pPos.Xd[0:3].copy())

        # 4- Enviar sinais de controle
        A.rSendControlSignals()

    if toc(tp) > 0.3:
        tp = tic()
        A.mCADplot(ax)
        RX = np.array(Rastro_X)
        RXd = np.array(Rastro_Xd)
        linhaX.set_data_3d(RX[:, 0], RX[:, 1], RX[:, 2])
        linhaXd.set_data_3d(RXd[:, 0], RXd[:, 1], RXd[:, 2])

        Info.set_text(f'Time: {toc(t):05.2f} | {tmax:05.2f}s')

        plt.pause(0.001)  # drawnow
    else:
        time.sleep(0.001)

# =========================================================================
print('Fim   -------------')
plt.ioff()

XX = np.array(XX)
tt = XX[:, -1]
# Colunas (MATLAB -> Python): Xd = 0..11, X = 12..23, Ud = 24..25, t = 26

# Posição
fig, axs = plt.subplots(3, 1)
fig.suptitle('Posição')
for i, (ax_i, nome) in enumerate(zip(axs, ['x', 'y', 'z'])):
    ax_i.plot(tt, XX[:, i], tt, XX[:, 12 + i])
    ax_i.legend([f'${nome}_{{Des}}$', f'${nome}_{{Atu}}$'])
    ax_i.grid(True)

# Orientação
fig, axs = plt.subplots(3, 1)
fig.suptitle('Orientação')
for i, (ax_i, nome) in enumerate(zip(axs, [r'\phi', r'\theta', r'\psi'])):
    ax_i.plot(tt, np.rad2deg(XX[:, 3 + i]), tt, np.rad2deg(XX[:, 15 + i]))
    ax_i.legend([f'${nome}_{{Des}}$', f'${nome}_{{Atu}}$'])
    ax_i.grid(True)

# Sinais de Controle
fig, axs = plt.subplots(2, 1)
fig.suptitle('Sinais de Controle')
axs[0].plot(tt, XX[:, 24])
axs[0].legend(['$u$ (m/s)'])
axs[0].grid(True)
axs[1].plot(tt, XX[:, 25])
axs[1].legend(['$\\omega$ (rad/s)'])
axs[1].grid(True)

plt.show()
