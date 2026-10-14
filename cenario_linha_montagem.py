"""
Transcrição de cenario_linha_montagem.m — cenário do Desafio 3.

Além da função de desenho, o arquivo exporta os dados do cenário para uso
nos programas dos alunos:
    OBSTACULOS -> blocos [x, y, largura, altura] (canto inferior esquerdo), em m
    ESTACOES   -> nome: (x, y, orientação de atendimento em graus)
    QUINAS     -> nome: (x, y) dos pontos de passagem do roteiro (curvas)

Os dados conferem com a tabela de dados de entrada do enunciado do Desafio 3.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# OBSTÁCULOS  [x y largura altura]
OBSTACULOS = [(-4, 1, 2, 3), (2, 1, 2, 3), (-4, -4, 2, 3), (2, -4, 2, 3)]
ALTURA_Z = 0.4   # altura dos blocos no desenho 3D (m)

# ESTAÇÕES: nome -> (x, y, orientação de atendimento em graus)
# A Base é o ponto de partida e de chegada. Cada bloco tem uma estação POR DENTRO
# (final 'i', corredor central) e uma POR FORA (final 'e', borda do cenário),
# ambas voltadas para o bloco.
ESTACOES = {
    'Base': (0.0, -5.0, 90),
    'SEe': (5.0, -2.5, 180),
    'NEe': (5.0, 2.5, 180),
    'NEi': (1.4, 2.5, 0),
    'SEi': (1.4, -2.5, 0),
    'SOi': (-1.4, -2.5, 180),
    'NOi': (-1.4, 2.5, 180),
    'NOe': (-5.0, 2.5, 0),
    'SOe': (-5.0, -2.5, 0),
}

# QUINAS (pontos de passagem do roteiro): nome -> (x, y)
QUINAS = {
    'C1': (5.0, -5.0), 'C2': (5.0, 5.0), 'C3': (1.4, 5.0), 'C4': (1.4, -4.4),
    'C5': (-1.4, -4.4), 'C6': (-1.4, 5.0), 'C7': (-5.0, 5.0), 'C8': (-5.0, -5.0),
}


def cenario_linha_montagem(ax=None):
    """Desenha o cenário no eixo ax. Funciona em eixo 3D (como no MATLAB) ou em
    eixo 2D comum (vista de cima, útil para os gráficos de trajetória)."""
    ax = ax if ax is not None else plt.gca()
    eixo3d = getattr(ax, 'name', '') == '3d'

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(True)
    if eixo3d:
        # axis equal + axis([-6 6 -6 6 0 2]) + view(-16,35)
        ax.set_zlim(0, 2)
        ax.set_zticks([0, 1, 2])
        ax.set_box_aspect((12, 12, 2))
        ax.view_init(elev=35, azim=-16 - 90)   # azimute do MATLAB -> matplotlib
        ax.computed_zorder = False             # desenha na ordem dos zorders
    else:
        ax.set_aspect('equal')
        ax.set_axisbelow(True)                 # grade atrás dos blocos
        ax.set_facecolor((0.7, 0.7, 0.7))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # CHÃO
    if eixo3d:
        chao = [[(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)]]
        ax.add_collection3d(Poly3DCollection(chao, facecolor=(0.7, 0.7, 0.7), zorder=0))

    # COORDENADAS
    if eixo3d:
        ax.quiver(0, 0, 0.05, 1, 0, 0, color='g', linewidth=2, arrow_length_ratio=0.3, zorder=3)
        ax.quiver(0, 0, 0.05, 0, 1, 0, color='g', linewidth=2, arrow_length_ratio=0.3, zorder=3)
        ax.text(1.1, 0, 0.07, 'X', color='green', fontsize=12)
        ax.text(0, 1.1, 0.07, 'Y', color='green', fontsize=12)
    else:
        ax.arrow(0, 0, 1, 0, color='g', width=0.03, head_width=0.15, length_includes_head=True)
        ax.arrow(0, 0, 0, 1, color='g', width=0.03, head_width=0.15, length_includes_head=True)
        ax.text(1.1, -0.1, 'X', color='green', fontsize=12)
        ax.text(-0.1, 1.1, 'Y', color='green', fontsize=12)

    # OBSTÁCULOS
    for x, y, w, h in OBSTACULOS:
        if eixo3d:
            xb = [x, x + w, x + w, x]
            yb = [y, y, y + h, y + h]
            topo = [list(zip(xb, yb, [ALTURA_Z] * 4))]
            ax.add_collection3d(Poly3DCollection(topo, facecolor='k', edgecolor='k', zorder=1))
            for j in range(4):
                k1, k2 = j, (j + 1) % 4
                lado = [[(xb[k1], yb[k1], 0), (xb[k2], yb[k2], 0),
                         (xb[k2], yb[k2], ALTURA_Z), (xb[k1], yb[k1], ALTURA_Z)]]
                ax.add_collection3d(Poly3DCollection(lado, facecolor=(0.3, 0.3, 0.3),
                                                     edgecolor='k', zorder=1))
        else:
            ax.add_patch(Rectangle((x, y), w, h, fc=(0.3, 0.3, 0.3), ec='k', zorder=1))

    # ESTAÇÕES (ponto vermelho + seta da orientação de atendimento)
    for nome, (px, py, ang) in ESTACOES.items():
        r = np.deg2rad(ang)
        if eixo3d:
            ax.plot([px], [py], [0.05], 'ro', markersize=6, zorder=4)
            ax.quiver(px, py, 0.05, 0.5 * np.cos(r), 0.5 * np.sin(r), 0, color='r',
                      linewidth=1.5, arrow_length_ratio=0.3, zorder=4)
            ax.text(px + 0.2, py + 0.15, 0.07, nome, color='red', fontsize=9)
        else:
            ax.plot(px, py, 'ro', markersize=6, zorder=4)
            ax.arrow(px, py, 0.5 * np.cos(r), 0.5 * np.sin(r), color='r', width=0.02,
                     head_width=0.12, length_includes_head=True, zorder=4)
            ax.text(px + 0.2, py + 0.15, nome, color='red', fontsize=9, weight='bold')

    # QUINAS (pontos de passagem)
    for nome, (px, py) in QUINAS.items():
        if eixo3d:
            ax.plot([px], [py], [0.05], 'b+', markersize=8, zorder=4)
            ax.text(px + 0.15, py - 0.35, 0.07, nome, color='blue', fontsize=8)
        else:
            ax.plot(px, py, 'b+', markersize=9, mew=1.5, zorder=4)
            ax.text(px + 0.15, py - 0.35, nome, color='blue', fontsize=8)


if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    cenario_linha_montagem(ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    cenario_linha_montagem(ax2)
    plt.show()
