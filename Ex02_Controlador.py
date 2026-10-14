"""Transcrição de Ex02_Controlador.m

Exemplo básico: leva o ponto de controle do Pioneer 3DX de (3, 3) até a origem.
A estrutura do laço (10 Hz) é a que deve ser usada no Desafio 3:
    rGetSensorData -> controlador -> pSC.Ud -> rSendControlSignals -> mCADplot -> LOG
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from pioneer3dx import Pioneer3DX, tic, toc

plt.close('all')

P = Pioneer3DX()
P.rSetPose(3, 3, 0)          # centro do eixo em (3, 3), voltado para +x

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(0, 3)
plt.ion()
plt.show()

tmax = 10    # Tempo em segundos
t = tic()    # Tempo corrente
tc = tic()   # Tempo de controle

XX = []

while toc(t) < tmax:
    if toc(tc) > 1 / 10:  # 10 Hz
        # ========================
        tc = tic()

        P.rGetSensorData()

        # Controlador ----
        a = P.pPar.a        # mesma distância usada pelo modelo do robô
        psi = P.pPos.X[5]
        K = np.array([[np.cos(psi), -a * np.sin(psi)],
                      [np.sin(psi),  a * np.cos(psi)]])
        U = np.linalg.solve(K, [-np.tanh(P.pPos.X[0]), -np.tanh(P.pPos.X[1])])

        P.pSC.Ud = U
        # ----------------
        P.rSendControlSignals()
        P.mCADplot(ax)
        plt.pause(0.001)

        # Armazenar dados: [x y psi u w t]
        XX.append(np.r_[P.pPos.X[[0, 1, 5]], P.pSC.Ud, toc(t)])
        # ========================
    else:
        time.sleep(0.001)   # evita ocupar 100% da CPU enquanto espera o próximo ciclo

XX = np.array(XX)
print(f'Posição final do ponto de controle: x = {XX[-1, 0]:.3f} m, y = {XX[-1, 1]:.3f} m')
plt.ioff()
plt.show()
