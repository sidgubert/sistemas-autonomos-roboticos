"""
Equivalente Python (simulação) da classe Pioneer3DX da plataforma AuRoRA.

Mantém a mesma organização de atributos do MATLAB:
    pPos.X   -> estado do ponto de controle [x y z phi theta psi dx dy dz dphi dtheta dpsi]
    pPos.Xd  -> estado desejado (mesmo formato)
    pPos.Xc  -> estado do centro do eixo das rodas (mesmo formato)
    pSC.Ud   -> sinais de controle desejados [u w] (m/s, rad/s)
    pSC.U    -> sinais efetivamente aplicados (depois da saturação)

O ponto de controle fica a pPar.a metros à frente do centro do eixo das rodas.

Observação: os índices em Python começam em 0, então X(1) no MATLAB é X[0],
X(6) (psi) é X[5], e assim por diante.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# --------------------------------------------------------------------------
# Equivalentes de tic / toc
def tic():
    return time.perf_counter()


def toc(t0):
    return time.perf_counter() - t0


class _Struct:
    """Estrutura simples, equivalente a um struct do MATLAB."""
    pass


# Faces de um paralelepípedo (índices dos 8 vértices)
_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
          [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


def _box(cx, cy, cz, lx, ly, lz):
    dx, dy, dz = lx / 2, ly / 2, lz / 2
    v = np.array([[-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
                  [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]])
    return v + np.array([cx, cy, cz])


class Pioneer3DX:
    def __init__(self, ID=1):
        self.pID = ID

        # Parâmetros
        self.pPar = _Struct()
        self.pPar.a = 0.15                     # distância do ponto de controle ao centro do eixo (m)
        self.pPar.Ts = 0.1                     # período de amostragem nominal (s) -> 10 Hz
        self.pPar.uSat = np.array([0.75, np.deg2rad(100)])  # saturação [m/s, rad/s]

        # Posições
        self.pPos = _Struct()
        self.pPos.X = np.zeros(12)
        self.pPos.Xd = np.zeros(12)
        self.pPos.Xc = np.zeros(12)
        self.pPos.Xtil = np.zeros(12)

        # Sinais de controle
        self.pSC = _Struct()
        self.pSC.U = np.zeros(2)
        self.pSC.Ud = np.zeros(2)
        self.pSC.Ur = np.zeros(2)

        self._t_last = None
        self._cad = None
        self._color = np.array([0.75, 0.0, 0.0])

    # ----------------------------------------------------------------------
    def rSetPose(self, x, y, psi):
        """Coloca o robô na pose (x, y, psi) do CENTRO DO EIXO das rodas (psi em rad)
        e atualiza o ponto de controle, que fica a pPar.a metros à frente."""
        psi = np.arctan2(np.sin(psi), np.cos(psi))
        self.pPos.Xc[:] = 0.0
        self.pPos.Xc[[0, 1, 5]] = [x, y, psi]
        self.pPos.X[:] = 0.0
        self.pPos.X[[0, 1]] = [x + self.pPar.a * np.cos(psi), y + self.pPar.a * np.sin(psi)]
        self.pPos.X[5] = psi
        self.pSC.U = np.zeros(2)
        self.pSC.Ud = np.zeros(2)
        self._t_last = None

    def rGetSensorData(self):
        """Em simulação, o estado já está atualizado pelo modelo cinemático.
        Recalcula o centro do eixo (Xc) a partir do ponto de controle (X)."""
        psi = self.pPos.X[5]
        self.pPos.Xc[[0, 1]] = self.pPos.X[[0, 1]] - self.pPar.a * np.array(
            [np.cos(psi), np.sin(psi)])
        self.pPos.Xc[5] = psi

    def rSendControlSignals(self, dt=None):
        """Aplica Ud ao modelo cinemático do robô (uniciclo) e integra a pose.

        dt = None : usa o tempo real decorrido desde a última chamada (como no robô
                    real); na primeira chamada usa pPar.Ts.
        dt = valor: usa esse passo fixo (simulação mais rápida que o tempo real e
                    resultados repetíveis).
        """
        now = time.perf_counter()
        if dt is None:
            dt = self.pPar.Ts if self._t_last is None else min(now - self._t_last, 0.5)
        self._t_last = now

        # Saturação dos atuadores (limites físicos do P3DX, componente a componente)
        U = np.clip(np.asarray(self.pSC.Ud, dtype=float).ravel(),
                    -self.pPar.uSat, self.pPar.uSat)
        self.pSC.U = U

        a = self.pPar.a
        X_old = self.pPos.X[[0, 1]].copy()
        psi = self.pPos.X[5]

        # Centro do eixo das rodas
        xc = self.pPos.X[[0, 1]] - a * np.array([np.cos(psi), np.sin(psi)])

        # Integração (Euler)
        xc = xc + U[0] * np.array([np.cos(psi), np.sin(psi)]) * dt
        psi = np.arctan2(np.sin(psi + U[1] * dt), np.cos(psi + U[1] * dt))

        # Atualiza centro do eixo e ponto de controle
        self.pPos.Xc[[0, 1]] = xc
        self.pPos.Xc[5] = psi
        self.pPos.Xc[[6, 7]] = U[0] * np.array([np.cos(psi), np.sin(psi)])
        self.pPos.Xc[11] = U[1]
        self.pPos.X[[0, 1]] = xc + a * np.array([np.cos(psi), np.sin(psi)])
        self.pPos.X[5] = psi
        if dt > 0:
            self.pPos.X[[6, 7]] = (self.pPos.X[[0, 1]] - X_old) / dt
        self.pPos.X[11] = U[1]

    # ----------------------------------------------------------------------
    def _robot_parts(self):
        """Corpo + 2 rodas no referencial do robô (origem no centro do eixo)."""
        return [
            (_box(-0.04, 0.0, 0.16, 0.45, 0.33, 0.16), 'body'),
            (_box(0.0, 0.20, 0.095, 0.19, 0.05, 0.19), 'wheel'),
            (_box(0.0, -0.20, 0.095, 0.19, 0.05, 0.19), 'wheel'),
        ]

    def _transformed_faces(self):
        psi = self.pPos.X[5]
        xc = self.pPos.X[[0, 1]] - self.pPar.a * np.array([np.cos(psi), np.sin(psi)])
        R = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])
        out = []
        for verts, kind in self._robot_parts():
            v = verts @ R.T + np.array([xc[0], xc[1], 0.0])
            out.append(([v[f] for f in _FACES], kind))
        return out

    def mCADplot(self, ax=None):
        """Desenha/atualiza o robô no eixo corrente (3D ou 2D, visto de cima)."""
        ax = ax if ax is not None else plt.gca()
        parts = self._transformed_faces()
        eixo3d = getattr(ax, 'name', '') == '3d'

        if self._cad is None or self._cad['ax'] is not ax:
            cols = []
            for faces, kind in parts:
                color = self._color if kind == 'body' else (0.1, 0.1, 0.1)
                if eixo3d:
                    pc = Poly3DCollection(faces, facecolor=color, edgecolor='k',
                                          linewidths=0.3, zorder=5)
                    ax.add_collection3d(pc)
                else:   # 2D: desenha só a face de baixo de cada peça (contorno visto de cima)
                    pc = Polygon(np.asarray(faces[0])[:, :2], closed=True, facecolor=color,
                                 edgecolor='k', linewidth=0.5, zorder=5)
                    ax.add_patch(pc)
                cols.append(pc)
            self._cad = {'ax': ax, 'cols': cols, '3d': eixo3d}
        else:
            for pc, (faces, _) in zip(self._cad['cols'], parts):
                if self._cad['3d']:
                    pc.set_verts(faces)
                else:
                    pc.set_xy(np.asarray(faces[0])[:, :2])

    def mCADcolor(self, color):
        """Muda a cor do corpo do robô (RGB em [0, 1])."""
        self._color = np.asarray(color, dtype=float)
        if self._cad is not None:
            self._cad['cols'][0].set_facecolor(self._color)
