"""
================================================================================
INSTITUTO FEDERAL DA BAHIA — CAMPUS VITÓRIA DA CONQUISTA
Disciplina: Robótica Terrestre
Avaliação: Navegação Contínua do Pioneer 3DX na Linha de Montagem
================================================================================

Este programa implementa as quatro tarefas da avaliação:
  - Tarefa A: Controle contínuo até uma pose em coordenadas polares (Siegwart / Aicardi).
  - Tarefa B: Roteiro contínuo entre blocos por seguidor de caminho (retas + arcos de raio 0.5 m)
              e demonstração do efeito da lei polar aplicada diretamente ao roteiro.
  - Tarefa C: Análise de robustez a ruído (odometria gaussiana + atuação multiplicativa, 10 execuções).
  - Tarefa D: Comparação quantitativa (com e sem ruído) entre três estratégias de navegação:
              1) Seguidor de Caminho (Tarefa B)
              2) Lei Polar ponto-a-ponto (Tarefa A com quinas como pontos de passagem)
              3) Controlador em Duas Fases (Gira-Anda-Gira)

Requisitos: numpy, matplotlib (sem dependências externas adicionais).
"""

import os
import sys
import time

# Garante suporte a UTF-8 em terminais Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc

from pioneer3dx import Pioneer3DX
from cenario_linha_montagem import cenario_linha_montagem, OBSTACULOS, ESTACOES, QUINAS

# ==============================================================================
# CONFIGURAÇÕES GERAIS E SELEÇÃO DE TAREFA (VARIÁVEIS DE TOPO)
# ==============================================================================
# Escolha a tarefa a executar: 'A', 'B', 'C', 'D' ou 'TODAS'
TAREFA = 'B'

# Habilita ruído de simulação (para Tarefa B e D; a Tarefa C força ruído = True)
HABILITAR_RUIDO = False

# Habilita animação visual em tempo real durante a execução (se False, simula em velocidade máxima)
ANIMACAO = False

# Salvar figuras geradas em arquivos png
SALVAR_GRAFICOS = True
PASTA_FIGURAS = 'figuras'

# Limites físicos e de controle do Pioneer 3DX
U_MAX = 0.75                  # m/s
W_MAX = np.deg2rad(100.0)     # rad/s (~1.7453 rad/s)
DT_NOMINAL = 0.1              # 10 Hz
MARGEM_SEGURANCA_OBS = 0.35   # m (distância mínima obrigatória aos obstáculos)


# ==============================================================================
# FUNÇÕES UTILITÁRIAS
# ==============================================================================
def wrap_angle(angle):
    """Normaliza o ângulo para o intervalo [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def saturate_ratio(u, w, u_max=U_MAX, w_max=W_MAX):
    """
    Aplica saturação aos sinais u e w preservando rigorosamente a razão u/w.
    Isso mantém o raio de curvatura instantâneo R = u/w inalterado.
    """
    su = abs(u) / u_max if u_max > 0 else 0.0
    sw = abs(w) / w_max if w_max > 0 else 0.0
    scale = max(1.0, su, sw)
    return u / scale, w / scale


def dist_to_obstacles(x, y):
    """
    Calcula a menor distância euclidiana do ponto (x, y) aos 4 obstáculos retangulares.
    """
    min_dist = 1e9
    for ox, oy, w, h in OBSTACULOS:
        dx = max(ox - x, 0.0, x - (ox + w))
        dy = max(oy - y, 0.0, y - (oy + h))
        d = np.sqrt(dx * dx + dy * dy)
        if d < min_dist:
            min_dist = d
    return min_dist


# ==============================================================================
# CONSTRUÇÃO DO CAMINHO DE REFERÊNCIA (RETAS E ARCOS DE RAIO 0.5 m)
# ==============================================================================
def construir_caminho_referencia(raio_arco=0.5):
    """
    Constrói a lista sequencial de segmentos do caminho de referência oficial
    (retas e arcos de circunferência tangentes de raio R = 0.5 m nas quinas C1 a C8).
    Total: 57.08 m (~57.1 m) conforme a Seção 3 do enunciado.
    """
    R = raio_arco
    segments = []

    def add_line(p1, p2, is_station=False, psi_f=None, name=""):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        dp = p2 - p1
        length = np.linalg.norm(dp)
        psi = np.arctan2(dp[1], dp[0])
        segments.append({
            'type': 'line', 'p1': p1, 'p2': p2, 'heading': psi, 'length': length,
            'is_station': is_station, 'psi_f': psi_f, 'name': name
        })

    def add_arc(center, r, t1, t2, ccw, name=""):
        center = np.array(center, dtype=float)
        dtheta = abs(t2 - t1)
        length = r * dtheta
        kappa = (1.0 / r) if ccw else (-1.0 / r)
        segments.append({
            'type': 'arc', 'center': center, 'r': r, 't1': t1, 't2': t2,
            'ccw': ccw, 'kappa': kappa, 'length': length,
            'is_station': False, 'psi_f': None, 'name': name
        })

    # 1. Base (0, -5) -> entrada C1 (5-R, -5)
    add_line((0.0, -5.0), (5.0 - R, -5.0), name="Base_C1")
    # 2. Quina C1 (Arco CCW, centro (5-R, -5+R))
    add_arc((5.0 - R, -5.0 + R), R, -np.pi / 2, 0.0, True, name="C1")
    # 3. Saída C1 até Estação SEe (5, -2.5)
    add_line((5.0, -5.0 + R), (5.0, -2.5), is_station=True, psi_f=np.deg2rad(180), name="SEe")
    # 4. Estação SEe até Estação NEe (5, 2.5)
    add_line((5.0, -2.5), (5.0, 2.5), is_station=True, psi_f=np.deg2rad(180), name="NEe")
    # 5. NEe até entrada C2 (5, 5-R)
    add_line((5.0, 2.5), (5.0, 5.0 - R), name="NEe_C2")
    # 6. Quina C2 (Arco CCW, centro (5-R, 5-R))
    add_arc((5.0 - R, 5.0 - R), R, 0.0, np.pi / 2, True, name="C2")
    # 7. Saída C2 até entrada C3 (1.4+R, 5)
    add_line((5.0 - R, 5.0), (1.4 + R, 5.0), name="C2_C3")
    # 8. Quina C3 (Arco CCW, centro (1.4+R, 5-R))
    add_arc((1.4 + R, 5.0 - R), R, np.pi / 2, np.pi, True, name="C3")
    # 9. Saída C3 até Estação NEi (1.4, 2.5)
    add_line((1.4, 5.0 - R), (1.4, 2.5), is_station=True, psi_f=np.deg2rad(0), name="NEi")
    # 10. Estação NEi até Estação SEi (1.4, -2.5)
    add_line((1.4, 2.5), (1.4, -2.5), is_station=True, psi_f=np.deg2rad(0), name="SEi")
    # 11. Estação SEi até entrada C4 (1.4, -4.4+R)
    add_line((1.4, -2.5), (1.4, -4.4 + R), name="SEi_C4")
    # 12. Quina C4 (Arco CW, centro (1.4-R, -4.4+R))
    add_arc((1.4 - R, -4.4 + R), R, 0.0, -np.pi / 2, False, name="C4")
    # 13. Saída C4 até entrada C5 (-1.4+R, -4.4)
    add_line((1.4 - R, -4.4), (-1.4 + R, -4.4), name="C4_C5")
    # 14. Quina C5 (Arco CW, centro (-1.4+R, -4.4+R))
    add_arc((-1.4 + R, -4.4 + R), R, -np.pi / 2, -np.pi, False, name="C5")
    # 15. Saída C5 até Estação SOi (-1.4, -2.5)
    add_line((-1.4, -4.4 + R), (-1.4, -2.5), is_station=True, psi_f=np.deg2rad(180), name="SOi")
    # 16. Estação SOi até Estação NOi (-1.4, 2.5)
    add_line((-1.4, -2.5), (-1.4, 2.5), is_station=True, psi_f=np.deg2rad(180), name="NOi")
    # 17. Estação NOi até entrada C6 (-1.4, 5-R)
    add_line((-1.4, 2.5), (-1.4, 5.0 - R), name="NOi_C6")
    # 18. Quina C6 (Arco CCW, centro (-1.4-R, 5-R))
    add_arc((-1.4 - R, 5.0 - R), R, 0.0, np.pi / 2, True, name="C6")
    # 19. Saída C6 até entrada C7 (-5+R, 5)
    add_line((-1.4 - R, 5.0), (-5.0 + R, 5.0), name="C6_C7")
    # 20. Quina C7 (Arco CCW, centro (-5+R, 5-R))
    add_arc((-5.0 + R, 5.0 - R), R, np.pi / 2, np.pi, True, name="C7")
    # 21. Saída C7 até Estação NOe (-5, 2.5)
    add_line((-5.0, 5.0 - R), (-5.0, 2.5), is_station=True, psi_f=np.deg2rad(0), name="NOe")
    # 22. Estação NOe até Estação SOe (-5, -2.5)
    add_line((-5.0, 2.5), (-5.0, -2.5), is_station=True, psi_f=np.deg2rad(0), name="SOe")
    # 23. Estação SOe até entrada C8 (-5, -5+R)
    add_line((-5.0, -2.5), (-5.0, -5.0 + R), name="SOe_C8")
    # 24. Quina C8 (Arco CCW, centro (-5+R, -5+R))
    add_arc((-5.0 + R, -5.0 + R), R, np.pi, 3.0 * np.pi / 2, True, name="C8")
    # 25. Saída C8 até Base final (0, -5)
    add_line((-5.0 + R, -5.0), (0.0, -5.0), is_station=True, psi_f=np.deg2rad(90), name="Base")

    return segments


def obter_pontos_caminho_referencia(segments, num_pontos=3000):
    """Gera uma lista densa de pontos (x, y) do caminho de referência para plotagem e avaliação."""
    pts = []
    total_L = sum(s['length'] for s in segments)
    for s in segments:
        n_seg = max(5, int(num_pontos * (s['length'] / total_L)))
        if s['type'] == 'line':
            x_line = np.linspace(s['p1'][0], s['p2'][0], n_seg)
            y_line = np.linspace(s['p1'][1], s['p2'][1], n_seg)
            for x, y in zip(x_line, y_line):
                pts.append((x, y))
        elif s['type'] == 'arc':
            c = s['center']
            r = s['r']
            thetas = np.linspace(s['t1'], s['t2'], n_seg)
            for th in thetas:
                pts.append((c[0] + r * np.cos(th), c[1] + r * np.sin(th)))
    return np.array(pts)


def calcular_desvio_caminho_referencia(xy_traj, ref_pts):
    """
    Calcula o maior desvio (máximo, em cm) e a média quadrática (RMS, em cm)
    de uma sequência de posições (x, y) em relação ao caminho de referência (retas + arcos).
    """
    xy = np.asarray(xy_traj, dtype=float)
    diffs = xy[:, None, :] - ref_pts[None, :, :]
    dists = np.hypot(diffs[:, :, 0], diffs[:, :, 1])
    min_d = np.min(dists, axis=1)
    max_cm = float(np.max(min_d) * 100.0)
    rms_cm = float(np.sqrt(np.mean(min_d ** 2)) * 100.0)
    return max_cm, rms_cm


def imprimir_tabela_metricas(titulo, tempo, comprimento, rho_cm, psi_deg, dev_max_cm, dev_rms_cm, folga_m, int_u2, int_w2):
    """Imprime no terminal a tabela padronizada das 6 métricas da Seção 5 do enunciado."""
    status_seguranca = "APROVADO (>= 0.35 m)" if folga_m >= MARGEM_SEGURANCA_OBS else "VIOLADO (< 0.35 m)!"
    print("\n" + "=" * 82)
    print(titulo)
    print("=" * 82)
    print(f"{'Métrica (Seção 5)':35s} | {'Valor Obtido':18s} | {'Unidade':8s} | {'Requisito':12s}")
    print("-" * 82)
    print(f"{'1. Tempo total':35s} | {tempo:18.1f} | {'s':8s} | {'<= 300 s':12s}")
    print(f"{'2. Comprimento percorrido':35s} | {comprimento:18.2f} | {'m':8s} | {'Ref: 57.1 m':12s}")
    print(f"{'3. Erro médio nas estações (rho)':35s} | {rho_cm:18.2f} | {'cm':8s} | {'< 3.0 cm':12s}")
    print(f"{'3. Erro médio nas estações (|dpsi|)':35s} | {psi_deg:18.2f} | {'°':8s} | {'< 2.0°':12s}")
    print(f"{'4. Desvio da referência (Máx.)':35s} | {dev_max_cm:18.2f} | {'cm':8s} | {'<= 5.0 cm':12s}")
    print(f"{'4. Desvio da referência (RMS)':35s} | {dev_rms_cm:18.2f} | {'cm':8s} | {'---':12s}")
    print(f"{'5. Folga mínima aos obstáculos':35s} | {folga_m:18.3f} | {'m':8s} | {status_seguranca:12s}")
    print(f"{'6. Esforço linear int(u^2 dt)':35s} | {int_u2:18.2f} | {'m^2/s':8s} | {'---':12s}")
    print(f"{'6. Esforço angular int(w^2 dt)':35s} | {int_w2:18.2f} | {'rad^2/s':8s} | {'---':12s}")
    print("=" * 82)


# ==============================================================================
# TAREFA A: CONTROLE CONTÍNUO ATÉ UMA POSE (LEI POLAR)
# ==============================================================================
def executar_tarefa_a(plotar=True, salvar=SALVAR_GRAFICOS):
    """
    Tarefa A — Controle contínuo até uma pose em coordenadas polares.
    Avalia 3 situações partindo de (0, 0, 0°):
      1) Destino à frente: (3, 1, 0°)
      2) Destino ao lado:   (0, 2, 90°)
      3) Destino atrás:     (-2, 0, 180°)
    """
    print("\n" + "=" * 75)
    print("EXECUTANDO TAREFA A — CONTROLE CONTÍNUO ATÉ UMA POSE (LEI POLAR)")
    print("=" * 75)

    casos = [
        {"nome": "Destino à Frente (3, 1, 0°)", "xd": 3.0, "yd": 1.0, "psif_deg": 0.0, "cor": "blue"},
        {"nome": "Destino ao Lado (0, 2, 90°)", "xd": 0.0, "yd": 2.0, "psif_deg": 90.0, "cor": "green"},
        {"nome": "Destino Atrás (-2, 0, 180°)", "xd": -2.0, "yd": 0.0, "psif_deg": 180.0, "cor": "red"}
    ]

    # Ganhos respeitando rigorosamente as condições de estabilidade de Lyapunov:
    #   kp > 0, kb < 0, ka - kp > 0
    kp = 0.50
    ka = 1.50   # ka - kp = 1.00 > 0
    kb = -0.30  # kb < 0

    dt = DT_NOMINAL
    historicos = {}

    print(f"{'Caso de Teste':30s} | {'Tempo':6s} | {'Comp.':7s} | {'Erro rho':9s} | {'Erro |dpsi|':11s} | {'int(u^2)':8s} | {'int(w^2)':8s} | {'Status'}")
    print("-" * 102)

    for caso in casos:
        nome = caso['nome']
        xd, yd = caso['xd'], caso['yd']
        psif = np.deg2rad(caso['psif_deg'])

        P = Pioneer3DX()
        P.rSetPose(0.0, 0.0, 0.0)

        t = 0.0
        t_max = 25.0
        log = []
        comp_total = 0.0
        prev_pos = P.pPos.Xc[[0, 1]].copy()
        int_u2 = 0.0
        int_w2 = 0.0

        arrived = False
        while t < t_max:
            P.rGetSensorData()
            xc, yc = P.pPos.Xc[[0, 1]]
            psi = P.pPos.Xc[5]

            dx = xd - xc
            dy = yd - yc
            rho = np.sqrt(dx * dx + dy * dy)

            # Critério de chegada: rho < 3 cm e |psi_f - psi| < 2 deg
            # Proteção contra a singularidade da lei polar em rho -> 0 (atan2(0,0) indefinido)
            if rho < 0.03:
                err_psi = wrap_angle(psif - psi)
                if abs(err_psi) < np.deg2rad(2.0):
                    u, w = 0.0, 0.0
                    arrived = True
                else:
                    u = 0.0
                    w = np.clip(3.0 * err_psi, -W_MAX, W_MAX)
            else:
                alpha = wrap_angle(np.arctan2(dy, dx) - psi)
                beta = wrap_angle(psif - psi - alpha)

                # Tratamento para |alpha| > 90°:
                # Se o alvo está atrás do robô, modular u por max(0, cos(alpha)) zera suavemente
                # o avanço frontal enquanto o robô gira em direção ao semiespaço do destino,
                # evitando laçadas amplas (cusps) na direção oposta ao alvo.
                u = kp * rho * max(0.0, np.cos(alpha))
                w = ka * alpha + kb * beta

            u_sat, w_sat = saturate_ratio(u, w)
            P.pSC.Ud = np.array([u_sat, w_sat])
            P.rSendControlSignals(dt=dt)

            cur_pos = P.pPos.Xc[[0, 1]].copy()
            comp_total += np.linalg.norm(cur_pos - prev_pos)
            prev_pos = cur_pos
            int_u2 += (u_sat ** 2) * dt
            int_w2 += (w_sat ** 2) * dt

            log.append([t, xc, yc, psi, rho, u_sat, w_sat])
            t += dt

            if arrived and t > 5.0:
                break

        log = np.array(log)
        pos_err = np.sqrt((xd - log[-1, 1])**2 + (yd - log[-1, 2])**2)
        ang_err = np.rad2deg(abs(wrap_angle(psif - log[-1, 3])))
        status = 'SUCESSO' if (pos_err < 0.03 and ang_err < 2.0) else 'FALHA'

        historicos[nome] = {
            'log': log, 'xd': xd, 'yd': yd, 'psif': psif,
            'pos_err': pos_err, 'ang_err': ang_err,
            'time': t, 'length': comp_total, 'int_u2': int_u2, 'int_w2': int_w2
        }

        print(f"{nome:30s} | {t:5.1f}s | {comp_total:5.2f} m | {pos_err*100:6.2f} cm | {ang_err:9.2f}° | {int_u2:8.2f} | {int_w2:8.2f} | {status}")

    if plotar:
        os.makedirs(PASTA_FIGURAS, exist_ok=True)

        # 1. Gráfico de Trajetórias
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal')
        ax.set_title("Tarefa A: Trajetórias sob Controle em Coordenadas Polares", fontsize=12, fontweight='bold')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        ax.plot(0, 0, 'ko', markersize=8, label="Partida (0, 0, 0°)")
        ax.quiver(0, 0, 0.4 * np.cos(0), 0.4 * np.sin(0), angles='xy', scale_units='xy', scale=1, color='k', width=0.007)

        for caso in casos:
            nome = caso['nome']
            d = historicos[nome]
            log = d['log']
            cor = caso['cor']
            ax.plot(log[:, 1], log[:, 2], label=f"{nome}", color=cor, linewidth=2)
            ax.plot(d['xd'], d['yd'], 'o', color=cor, markersize=7)
            ax.quiver(d['xd'], d['yd'], 0.4 * np.cos(d['psif']), 0.4 * np.sin(d['psif']),
                      angles='xy', scale_units='xy', scale=1, color=cor, width=0.006)

        ax.set_xlim(-3, 4)
        ax.set_ylim(-1.5, 3)
        ax.legend(loc='best')
        if salvar:
            fig.savefig(os.path.join(PASTA_FIGURAS, "trajetoria_tarefa_A.png"), dpi=300, bbox_inches='tight')

        # 2. Gráficos de Erros e Sinais de Controle
        fig2, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        fig2.suptitle("Tarefa A: Sinais de Controle e Distância ao Alvo", fontsize=12, fontweight='bold')

        for caso in casos:
            nome = caso['nome']
            d = historicos[nome]
            log = d['log']
            cor = caso['cor']
            t_axis = log[:, 0]
            axs[0].plot(t_axis, log[:, 4] * 100, label=nome, color=cor, linewidth=1.8)
            axs[1].plot(t_axis, log[:, 5], label=nome, color=cor, linewidth=1.8)
            axs[2].plot(t_axis, np.rad2deg(log[:, 6]), label=nome, color=cor, linewidth=1.8)

        axs[0].set_ylabel("Distância $\\rho$ (cm)")
        axs[0].axhline(3.0, color='r', linestyle=':', label='Tolerância (3 cm)')
        axs[0].grid(True)
        axs[0].legend(loc='upper right')

        axs[1].set_ylabel("Velocidade $u$ (m/s)")
        axs[1].axhline(U_MAX, color='k', linestyle='--', alpha=0.5)
        axs[1].grid(True)

        axs[2].set_ylabel("Vel. Angular $\\omega$ (°/s)")
        axs[2].set_xlabel("Tempo (s)")
        axs[2].axhline(np.rad2deg(W_MAX), color='k', linestyle='--', alpha=0.5)
        axs[2].axhline(-np.rad2deg(W_MAX), color='k', linestyle='--', alpha=0.5)
        axs[2].grid(True)

        if salvar:
            fig2.savefig(os.path.join(PASTA_FIGURAS, "sinais_tarefa_A.png"), dpi=300, bbox_inches='tight')
        plt.show(block=False)

    return historicos


# ==============================================================================
# TAREFA B: SEGUIDOR DE CAMINHO NA LINHA DE MONTAGEM
# ==============================================================================
def executar_seguidor_caminho(ruido=False, semente=None, animar=ANIMACAO, plotar=True, salvar=SALVAR_GRAFICOS):
    """
    Tarefa B / Núcleo do seguidor de caminho:
      - Roteiro oficial de 17 destinos (Base -> C1..C8 -> 8 estações -> Base)
      - Retas e arcos tangentes de raio R = 0.5 m nas quinas
      - Transição contínua sem parada nas quinas
      - Parada nas estações (rho < 3 cm, |psi_f - psi| < 2 deg)
      - Lei de controle:
          u = clip(u_ref * max(cos(e_psi), 0), -U_MAX, U_MAX)
          w = u * kappa + k_psi * e_psi - k_y * u * e_y
    """
    rng = np.random.default_rng(semente) if ruido else None
    segments = construir_caminho_referencia(raio_arco=0.5)

    P = Pioneer3DX()
    # Parte da Base em (0, -5), psi = 90°
    P.rSetPose(0.0, -5.0, np.deg2rad(90.0))

    dt = DT_NOMINAL
    t = 0.0
    t_max = 300.0

    seg_idx = 0
    state = 'ALIGN_START'
    wait_counter = 0

    # Ganhos do seguidor de caminho
    u_nom = 0.42       # m/s (velocidade nominal em trechos retos)
    u_arc = 0.32       # m/s (velocidade em curvas para preservar raio)
    k_psi = 5.0        # ganho de erro de rumo
    k_y = 10.0         # ganho de erro lateral

    log = []
    station_metrics = []

    # Métricas acumuladas
    comprimento_total = 0.0
    prev_pos = P.pPos.Xc[[0, 1]].copy()
    folga_minima = 1e9
    maior_desvio = 0.0
    soma_ey2 = 0.0
    cont_ey = 0
    esforco_u2 = 0.0
    esforco_w2 = 0.0

    # Configuração de visualização animada (se solicitada)
    ax_anim = None
    if animar:
        plt.ion()
        fig_anim = plt.figure(figsize=(9, 8))
        ax_anim = fig_anim.add_subplot(1, 1, 1)
        cenario_linha_montagem(ax_anim)
        P.mCADplot(ax_anim)
        ref_pts = obter_pontos_caminho_referencia(segments)
        ax_anim.plot(ref_pts[:, 0], ref_pts[:, 1], 'm--', linewidth=1.5, alpha=0.7, label="Caminho Ref.")
        rastro_x, rastro_y = [P.pPos.Xc[0]], [P.pPos.Xc[1]]
        linha_rastro, = ax_anim.plot(rastro_x, rastro_y, 'b-', linewidth=1.8, label="P3DX")
        title_text = ax_anim.set_title(f"Simulação P3DX — Tempo: 0.0s")
        plt.show()

    while t < t_max and seg_idx < len(segments):
        # 1. Leitura de sensores (Pose Verdadeira)
        P.rGetSensorData()
        xc_true, yc_true = P.pPos.Xc[[0, 1]]
        psi_true = P.pPos.Xc[5]

        # Injeção de ruído na odometria (se habilitado)
        if ruido:
            xc_ctrl = xc_true + rng.normal(0, 0.01)
            yc_ctrl = yc_true + rng.normal(0, 0.01)
            psi_ctrl = wrap_angle(psi_true + rng.normal(0, np.deg2rad(0.5)))
        else:
            xc_ctrl, yc_ctrl, psi_ctrl = xc_true, yc_true, psi_true

        cur_seg = segments[seg_idx]
        u = 0.0
        w = 0.0
        e_y_ctrl = 0.0
        e_psi_ctrl = 0.0

        # 2. Controlador (Máquina de estados para navegação contínua e paradas nas estações)
        if state == 'ALIGN_START':
            # Alinhamento inicial da Base para o primeiro trecho (+x)
            target_psi = segments[0]['heading']
            err_psi = wrap_angle(target_psi - psi_ctrl)
            if abs(err_psi) < np.deg2rad(1.0):
                state = 'FOLLOW'
            else:
                w = np.clip(3.0 * err_psi, -W_MAX, W_MAX)

        elif state == 'FOLLOW':
            if cur_seg['type'] == 'line':
                p1 = cur_seg['p1']
                p2 = cur_seg['p2']
                heading = cur_seg['heading']
                L = cur_seg['length']
                tangent = np.array([np.cos(heading), np.sin(heading)])
                normal = np.array([-np.sin(heading), np.cos(heading)])

                v = np.array([xc_ctrl, yc_ctrl]) - p1
                s = np.dot(v, tangent)
                e_y_ctrl = np.dot(v, normal)
                e_psi_ctrl = wrap_angle(heading - psi_ctrl)
                kappa = 0.0

                rem_dist = L - s
                if cur_seg['is_station'] and rem_dist <= 0.35:
                    # Desaceleração suave ao se aproximar da estação
                    dist_to_end = np.linalg.norm(np.array([xc_ctrl, yc_ctrl]) - p2)
                    u_ref = max(0.04, u_nom * (dist_to_end / 0.40))
                    if dist_to_end < 0.015 or rem_dist <= 0.005:
                        state = 'STOP_AT_STATION'
                        u, w = 0.0, 0.0
                    else:
                        # Limita u antes de calcular w para manter consistência com u*kappa e -k_y*u*e_y
                        u = np.clip(u_ref * max(np.cos(e_psi_ctrl), 0.0), -U_MAX, U_MAX)
                        w = u * kappa + k_psi * e_psi_ctrl - k_y * u * e_y_ctrl
                elif not cur_seg['is_station'] and rem_dist <= 0.03:
                    # Transição imediata e suave para a próxima quina/curva sem parada
                    seg_idx += 1
                    continue
                else:
                    u = np.clip(u_nom * max(np.cos(e_psi_ctrl), 0.0), -U_MAX, U_MAX)
                    w = u * kappa + k_psi * e_psi_ctrl - k_y * u * e_y_ctrl

            elif cur_seg['type'] == 'arc':
                center = cur_seg['center']
                r = cur_seg['r']
                t1 = cur_seg['t1']
                t2 = cur_seg['t2']
                ccw = cur_seg['ccw']
                kappa = cur_seg['kappa']
                L = cur_seg['length']

                v = np.array([xc_ctrl, yc_ctrl]) - center
                dist_center = np.linalg.norm(v)
                angle_cur = np.arctan2(v[1], v[0])

                if ccw:
                    heading = angle_cur + np.pi / 2.0
                    e_y_ctrl = r - dist_center
                    d_theta = (angle_cur - t1) % (2.0 * np.pi)
                    arc_progress = 0.0 if d_theta > np.pi else r * d_theta
                else:
                    heading = angle_cur - np.pi / 2.0
                    e_y_ctrl = dist_center - r
                    d_theta = (t1 - angle_cur) % (2.0 * np.pi)
                    arc_progress = 0.0 if d_theta > np.pi else r * d_theta

                e_psi_ctrl = wrap_angle(heading - psi_ctrl)
                rem_dist = L - arc_progress
                if rem_dist <= 0.03:
                    # Transição da curva para a reta tangente subsequente sem parar
                    seg_idx += 1
                    continue
                else:
                    u = np.clip(u_arc * max(np.cos(e_psi_ctrl), 0.0), -U_MAX, U_MAX)
                    w = u * kappa + k_psi * e_psi_ctrl - k_y * u * e_y_ctrl

        elif state == 'STOP_AT_STATION':
            # Gira no lugar para alinhar com a orientação de atendimento psi_f
            psi_f = cur_seg['psi_f']
            err_psi = wrap_angle(psi_f - psi_ctrl)
            dist_to_target_true = np.linalg.norm(np.array([xc_true, yc_true]) - cur_seg['p2'])
            err_psi_true = abs(wrap_angle(psi_f - psi_true))

            if abs(err_psi) < np.deg2rad(1.0):
                station_metrics.append({
                    'name': cur_seg['name'],
                    'rho_cm': dist_to_target_true * 100.0,
                    'psi_err_deg': np.rad2deg(err_psi_true),
                    'time': t
                })
                state = 'WAIT_STATION'
                wait_counter = 5  # 0.5s de atendimento parado
                u, w = 0.0, 0.0
            else:
                u = 0.0
                w = np.clip(3.0 * err_psi, -np.deg2rad(80), np.deg2rad(80))

        elif state == 'WAIT_STATION':
            u, w = 0.0, 0.0
            wait_counter -= 1
            if wait_counter <= 0:
                if seg_idx == len(segments) - 1:
                    state = 'DONE'
                    break
                else:
                    seg_idx += 1
                    state = 'REALIGN_PATH'

        elif state == 'REALIGN_PATH':
            # Reorienta em direção à reta seguinte
            next_heading = segments[seg_idx]['heading']
            err_psi = wrap_angle(next_heading - psi_ctrl)
            if abs(err_psi) < np.deg2rad(1.0):
                state = 'FOLLOW'
            else:
                u = 0.0
                w = np.clip(3.0 * err_psi, -np.deg2rad(80), np.deg2rad(80))

        # Saturação no controlador preservando a razão u/w
        u_sat, w_sat = saturate_ratio(u, w)

        # Injeção de ruído de atuação multiplicativo: u e w * (1 + N(0, 0.05))
        if ruido:
            u_act = u_sat * (1.0 + rng.normal(0, 0.05))
            w_act = w_sat * (1.0 + rng.normal(0, 0.05))
            u_applied, w_applied = saturate_ratio(u_act, w_act)
        else:
            u_applied, w_applied = u_sat, w_sat

        # 3. Enviar sinais de controle (pSC.Ud -> rSendControlSignals)
        P.pSC.Ud = np.array([u_applied, w_applied])
        P.rSendControlSignals(dt=dt)
        t += dt

        # 4. Métricas calculadas com a POSE VERDADEIRA do simulador
        cur_pos = P.pPos.Xc[[0, 1]].copy()
        step_len = np.linalg.norm(cur_pos - prev_pos)
        comprimento_total += step_len
        prev_pos = cur_pos

        d_obs = dist_to_obstacles(xc_true, yc_true)
        if d_obs < folga_minima:
            folga_minima = d_obs

        # Erro lateral verdadeiro em relação ao segmento ativo
        if state == 'FOLLOW':
            if cur_seg['type'] == 'line':
                v_true = np.array([xc_true, yc_true]) - cur_seg['p1']
                norm_true = np.array([-np.sin(cur_seg['heading']), np.cos(cur_seg['heading'])])
                ey_true = np.dot(v_true, norm_true)
                epsi_true = wrap_angle(cur_seg['heading'] - psi_true)
            elif cur_seg['type'] == 'arc':
                d_c = np.linalg.norm(np.array([xc_true, yc_true]) - cur_seg['center'])
                ey_true = cur_seg['r'] - d_c if cur_seg['ccw'] else d_c - cur_seg['r']
                angle_true = np.arctan2(yc_true - cur_seg['center'][1], xc_true - cur_seg['center'][0])
                head_arc = angle_true + (np.pi / 2 if cur_seg['ccw'] else -np.pi / 2)
                epsi_true = wrap_angle(head_arc - psi_true)

            if abs(ey_true) > maior_desvio:
                maior_desvio = abs(ey_true)
            soma_ey2 += (ey_true ** 2)
            cont_ey += 1
        else:
            ey_true = 0.0
            epsi_true = 0.0

        esforco_u2 += (u_applied ** 2) * dt
        esforco_w2 += (w_applied ** 2) * dt

        log.append([t, xc_true, yc_true, psi_true, xc_ctrl, yc_ctrl, psi_ctrl,
                    u_applied, w_applied, ey_true, epsi_true, state])

        # 5. Plotagem mCADplot (quando animação ativa)
        if animar and len(log) % 3 == 0:
            P.mCADplot(ax_anim)
            rastro_x.append(xc_true)
            rastro_y.append(yc_true)
            linha_rastro.set_data(rastro_x, rastro_y)
            title_text.set_text(f"P3DX Linha de Montagem | Tempo: {t:05.1f}s / {t_max:05.1f}s | Estado: {state}")
            plt.pause(0.001)

    if animar:
        plt.ioff()

    rms_desvio = np.sqrt(soma_ey2 / max(1, cont_ey))
    log = np.array(log, dtype=object)

    resultado = {
        'tempo_total': t,
        'comprimento': comprimento_total,
        'maior_desvio_cm': maior_desvio * 100.0,
        'rms_desvio_cm': rms_desvio * 100.0,
        'folga_minima': folga_minima,
        'esforco_u2': esforco_u2,
        'esforco_w2': esforco_w2,
        'estacoes': station_metrics,
        'log': log,
        'segments': segments
    }

    # Gráficos da Tarefa B
    if plotar:
        os.makedirs(PASTA_FIGURAS, exist_ok=True)
        # Executa também a lei polar no roteiro para demonstrar as ondulações conforme pedido na Tarefa B
        res_polar_b = executar_estrategia_polar(ruido=ruido, semente=semente if semente is not None else 42)

        # 1. Trajetória com Cenário Completo (mostrando Seguidor de Caminho + efeito da Lei Polar)
        fig, ax = plt.subplots(figsize=(10, 9))
        cenario_linha_montagem(ax)
        ref_pts = obter_pontos_caminho_referencia(segments)
        ax.plot(ref_pts[:, 0], ref_pts[:, 1], 'm--', linewidth=1.5, label="Caminho Ref. (Retas + Arcos R=0.5m)", zorder=3)
        ax.plot(res_polar_b['log'][:, 1], res_polar_b['log'][:, 2], 'r:', linewidth=1.6, alpha=0.75,
                label="Lei Polar Direta no Roteiro (Ondulações)", zorder=4)
        ax.plot(log[:, 1], log[:, 2], 'b-', linewidth=2.1, label="Seguidor de Caminho P3DX (Tarefa B)", zorder=5)
        ax.set_title(f"Tarefa B: Roteiro Completo na Linha de Montagem (Tempo: {t:.1f} s)", fontsize=12, fontweight='bold')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        if salvar:
            fig.savefig(os.path.join(PASTA_FIGURAS, "trajetoria_tarefa_B.png"), dpi=300, bbox_inches='tight')

        # 2. Erros Lateral e de Rumo
        t_arr = np.array(log[:, 0], dtype=float)
        ey_arr = np.array(log[:, 9], dtype=float) * 100.0
        epsi_arr = np.rad2deg(np.array(log[:, 10], dtype=float))

        fig_err, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        fig_err.suptitle("Tarefa B: Erros de Rastreamento ao Longo do Tempo", fontsize=12, fontweight='bold')
        axs[0].plot(t_arr, ey_arr, 'r-', linewidth=1.5, label="Erro Lateral $e_y$")
        axs[0].axhline(5.0, color='k', linestyle=':', label="Limite Máximo (+5 cm)")
        axs[0].axhline(-5.0, color='k', linestyle=':', label="Limite Mínimo (-5 cm)")
        axs[0].set_ylabel("Erro Lateral (cm)")
        axs[0].grid(True)
        axs[0].legend(loc='upper right')

        axs[1].plot(t_arr, epsi_arr, 'g-', linewidth=1.5, label="Erro de Rumo $e_\\psi$")
        axs[1].set_ylabel("Erro de Rumo (°)")
        axs[1].set_xlabel("Tempo (s)")
        axs[1].grid(True)
        axs[1].legend(loc='upper right')
        if salvar:
            fig_err.savefig(os.path.join(PASTA_FIGURAS, "erros_tarefa_B.png"), dpi=300, bbox_inches='tight')

        # 3. Sinais de Controle
        u_arr = np.array(log[:, 7], dtype=float)
        w_arr = np.rad2deg(np.array(log[:, 8], dtype=float))

        fig_sig, axs_sig = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        fig_sig.suptitle("Tarefa B: Sinais de Controle com Saturação Preservando u/w", fontsize=12, fontweight='bold')
        axs_sig[0].plot(t_arr, u_arr, 'b-', linewidth=1.5)
        axs_sig[0].axhline(U_MAX, color='k', linestyle='--', label=f"Saturação ({U_MAX} m/s)")
        axs_sig[0].set_ylabel("Velocidade $u$ (m/s)")
        axs_sig[0].grid(True)
        axs_sig[0].legend(loc='upper right')

        axs_sig[1].plot(t_arr, w_arr, 'm-', linewidth=1.5)
        axs_sig[1].axhline(np.rad2deg(W_MAX), color='k', linestyle='--', label="Saturação (±100°/s)")
        axs_sig[1].axhline(-np.rad2deg(W_MAX), color='k', linestyle='--')
        axs_sig[1].set_ylabel("Vel. Angular $\\omega$ (°/s)")
        axs_sig[1].set_xlabel("Tempo (s)")
        axs_sig[1].grid(True)
        axs_sig[1].legend(loc='upper right')
        if salvar:
            fig_sig.savefig(os.path.join(PASTA_FIGURAS, "sinais_tarefa_B.png"), dpi=300, bbox_inches='tight')

        plt.show(block=False)

    return resultado


# ==============================================================================
# TAREFA C: AVALIAÇÃO DE ROBUSTEZ A RUÍDO (10 EXECUÇÕES)
# ==============================================================================
def executar_tarefa_c(num_execucoes=10, salvar=SALVAR_GRAFICOS):
    """
    Tarefa C — Robustez a ruído.
    Executa o roteiro completo 10 vezes com sementes aleatórias distintas,
    utilizando ruído gaussiano na odometria e multiplicativo na atuação.
    Calcula média e desvio padrão para todas as 6 métricas oficiais (com pose verdadeira).
    """
    print("\n" + "=" * 75)
    print(f"EXECUTANDO TAREFA C — ROBUSTEZ A RUÍDO ({num_execucoes} EXECUÇÕES)")
    print("=" * 75)

    resultados = []
    for i in range(num_execucoes):
        semente = 100 + i
        print(f"Execução {i + 1:2d}/{num_execucoes} (Semente: {semente})...", end="", flush=True)
        res = executar_seguidor_caminho(ruido=True, semente=semente, animar=False, plotar=False)
        resultados.append(res)
        print(f" Concluído em {res['tempo_total']:.1f}s | Desvio Máx: {res['maior_desvio_cm']:.2f} cm | Folga Mín: {res['folga_minima']:.3f} m")

    # Consolidação estatística
    tempos = [r['tempo_total'] for r in resultados]
    comprimentos = [r['comprimento'] for r in resultados]
    desvios_max = [r['maior_desvio_cm'] for r in resultados]
    desvios_rms = [r['rms_desvio_cm'] for r in resultados]
    folgas = [r['folga_minima'] for r in resultados]
    esforcos_u = [r['esforco_u2'] for r in resultados]
    esforcos_w = [r['esforco_w2'] for r in resultados]

    erros_pos_estacoes = [np.mean([s['rho_cm'] for s in r['estacoes']]) for r in resultados]
    erros_ang_estacoes = [np.mean([s['psi_err_deg'] for s in r['estacoes']]) for r in resultados]

    print("\n" + "=" * 88)
    print("TABELA DE MÉTRICAS DA TAREFA C (MÉDIA ± DESVIO PADRÃO, 10 EXECUÇÕES)")
    print("=" * 88)
    metricas_c = [
        ("1. Tempo total", np.mean(tempos), np.std(tempos), "s", "<= 300 s"),
        ("2. Comprimento percorrido", np.mean(comprimentos), np.std(comprimentos), "m", "Ref: ~57.1 m"),
        ("3. Erro pose estações (rho)", np.mean(erros_pos_estacoes), np.std(erros_pos_estacoes), "cm", "< 3.0 cm"),
        ("3. Erro pose estações (|dpsi|)", np.mean(erros_ang_estacoes), np.std(erros_ang_estacoes), "deg", "< 2.0 deg"),
        ("4. Desvio da referência (Máx.)", np.mean(desvios_max), np.std(desvios_max), "cm", "<= 5.0 cm"),
        ("4. Desvio da referência (RMS)", np.mean(desvios_rms), np.std(desvios_rms), "cm", "---"),
        ("5. Folga mínima a obstáculos", np.mean(folgas), np.std(folgas), "m", ">= 0.35 m"),
        ("6. Esforço de controle (u^2)", np.mean(esforcos_u), np.std(esforcos_u), "m^2/s", "---"),
        ("6. Esforço de controle (w^2)", np.mean(esforcos_w), np.std(esforcos_w), "rad^2/s", "---"),
    ]

    print(f"{'Métrica':32s} | {'Valor (Média ± Desvio Padrão)':30s} | {'Unidade':7s} | {'Exigência':12s}")
    print("-" * 88)
    for nome, m, s, unid, req in metricas_c:
        val_str = f"{m:.3f} ± {s:.3f}"
        print(f"{nome:32s} | {val_str:30s} | {unid:7s} | {req:12s}")
    print("=" * 88)

    if salvar:
        os.makedirs(PASTA_FIGURAS, exist_ok=True)
        res_ex = resultados[0]
        log = res_ex['log']
        fig, ax = plt.subplots(figsize=(10, 9))
        cenario_linha_montagem(ax)
        ref_pts = obter_pontos_caminho_referencia(res_ex['segments'])
        ax.plot(ref_pts[:, 0], ref_pts[:, 1], 'm--', linewidth=1.5, label="Referência", zorder=3)
        ax.plot(log[:, 1], log[:, 2], 'b-', linewidth=2.0, label="Pose Verdadeira", zorder=5)
        ax.plot(log[:, 4], log[:, 5], 'r.', markersize=1.2, alpha=0.45, label="Odometria Ruidosa", zorder=4)
        ax.set_title("Tarefa C: Trajetória com Ruído de Odometria e Atuação", fontsize=12, fontweight='bold')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3)
        fig.savefig(os.path.join(PASTA_FIGURAS, "trajetoria_tarefa_C.png"), dpi=300, bbox_inches='tight')
        plt.show(block=False)

    return metricas_c


# ==============================================================================
# TAREFA D: COMPARAÇÃO QUANTITATIVA DE TRÊS ESTRATÉGIAS (COM E SEM RUÍDO)
# ==============================================================================
WAYPOINTS_ROTA = [
    ('C1', 5.0, -5.0, 90.0, False),
    ('SEe', 5.0, -2.5, 180.0, True),
    ('NEe', 5.0, 2.5, 180.0, True),
    ('C2', 5.0, 5.0, 180.0, False),
    ('C3', 1.4, 5.0, -90.0, False),
    ('NEi', 1.4, 2.5, 0.0, True),
    ('SEi', 1.4, -2.5, 0.0, True),
    ('C4', 1.4, -4.4, 180.0, False),
    ('C5', -1.4, -4.4, 90.0, False),
    ('SOi', -1.4, -2.5, 180.0, True),
    ('NOi', -1.4, 2.5, 180.0, True),
    ('C6', -1.4, 5.0, 180.0, False),
    ('C7', -5.0, 5.0, -90.0, False),
    ('NOe', -5.0, 2.5, 0.0, True),
    ('SOe', -5.0, -2.5, 0.0, True),
    ('C8', -5.0, -5.0, 0.0, False),
    ('Base', 0.0, -5.0, 90.0, True),
]


def executar_estrategia_polar(ruido=False, semente=42):
    """Estratégia 2: Lei Polar da Tarefa A aplicada diretamente aos pontos da rota."""
    rng = np.random.default_rng(semente) if ruido else None
    segments = construir_caminho_referencia(raio_arco=0.5)
    ref_pts = obter_pontos_caminho_referencia(segments, num_pontos=3000)

    P = Pioneer3DX()
    P.rSetPose(0.0, -5.0, np.deg2rad(90.0))
    dt = DT_NOMINAL
    t = 0.0
    wp_idx = 0
    state = 'POLAR'
    wait_counter = 0

    total_len = 0.0
    prev_pos = P.pPos.Xc[[0, 1]].copy()
    folga_min = 1e9
    int_u2 = 0.0
    int_w2 = 0.0
    station_errs = []
    log = []

    kp, ka, kb = 0.40, 1.20, -0.30

    while t < 300.0 and wp_idx < len(WAYPOINTS_ROTA):
        P.rGetSensorData()
        xc_true, yc_true = P.pPos.Xc[[0, 1]]
        psi_true = P.pPos.Xc[5]

        if ruido:
            xc = xc_true + rng.normal(0, 0.01)
            yc = yc_true + rng.normal(0, 0.01)
            psi = wrap_angle(psi_true + rng.normal(0, np.deg2rad(0.5)))
        else:
            xc, yc, psi = xc_true, yc_true, psi_true

        wp = WAYPOINTS_ROTA[wp_idx]
        name, tx, ty, tpsi_deg, is_stop = wp
        tpsi = np.deg2rad(tpsi_deg)

        dx = tx - xc
        dy = ty - yc
        rho = np.sqrt(dx * dx + dy * dy)
        u, w = 0.0, 0.0

        if state == 'POLAR':
            threshold = 0.03 if is_stop else 0.25
            if rho < threshold:
                if is_stop:
                    state = 'ALIGN_FINAL'
                else:
                    wp_idx += 1
                    continue
            else:
                alpha = wrap_angle(np.arctan2(dy, dx) - psi)
                beta = wrap_angle(tpsi - psi - alpha)
                u = kp * rho * max(0.0, np.cos(alpha))
                w = ka * alpha + kb * beta

        elif state == 'ALIGN_FINAL':
            err_final = wrap_angle(tpsi - psi)
            dist_true = np.linalg.norm(np.array([xc_true, yc_true]) - np.array([tx, ty]))
            if abs(err_final) < np.deg2rad(1.5):
                station_errs.append({
                    'name': name,
                    'rho_cm': dist_true * 100.0,
                    'psi_err_deg': np.rad2deg(abs(wrap_angle(tpsi - psi_true)))
                })
                state = 'WAIT'
                wait_counter = 5
            else:
                w = np.clip(3.0 * err_final, -W_MAX, W_MAX)

        elif state == 'WAIT':
            wait_counter -= 1
            if wait_counter <= 0:
                wp_idx += 1
                if wp_idx >= len(WAYPOINTS_ROTA):
                    break
                state = 'POLAR'

        u_sat, w_sat = saturate_ratio(u, w)
        if ruido:
            u_act = u_sat * (1.0 + rng.normal(0, 0.05))
            w_act = w_sat * (1.0 + rng.normal(0, 0.05))
            u_app, w_app = saturate_ratio(u_act, w_act)
        else:
            u_app, w_app = u_sat, w_sat

        P.pSC.Ud = np.array([u_app, w_app])
        P.rSendControlSignals(dt=dt)
        t += dt

        cur_pos = P.pPos.Xc[[0, 1]].copy()
        total_len += np.linalg.norm(cur_pos - prev_pos)
        prev_pos = cur_pos

        d_obs = dist_to_obstacles(xc_true, yc_true)
        if d_obs < folga_min:
            folga_min = d_obs
        int_u2 += (u_app ** 2) * dt
        int_w2 += (w_app ** 2) * dt

        log.append([t, xc_true, yc_true, psi_true, u_app, w_app])

    log_arr = np.array(log)
    dev_max_cm, dev_rms_cm = calcular_desvio_caminho_referencia(log_arr[:, 1:3], ref_pts)

    return {
        'nome': 'Lei Polar (Ponto a Ponto)',
        'tempo': t,
        'dist': total_len,
        'maior_desvio_cm': dev_max_cm,
        'rms_desvio_cm': dev_rms_cm,
        'min_obs': folga_min,
        'int_u2': int_u2,
        'int_w2': int_w2,
        'mean_rho': np.mean([s['rho_cm'] for s in station_errs]) if station_errs else 0.0,
        'mean_psi': np.mean([s['psi_err_deg'] for s in station_errs]) if station_errs else 0.0,
        'estacoes': station_errs,
        'log': log_arr
    }


def executar_estrategia_duas_fases(ruido=False, semente=42):
    """Estratégia 3: Controlador em Duas Fases (Gira-Anda-Gira)."""
    rng = np.random.default_rng(semente) if ruido else None
    segments = construir_caminho_referencia(raio_arco=0.5)
    ref_pts = obter_pontos_caminho_referencia(segments, num_pontos=3000)

    P = Pioneer3DX()
    P.rSetPose(0.0, -5.0, np.deg2rad(90.0))
    dt = DT_NOMINAL
    t = 0.0
    wp_idx = 0
    phase = 'TURN_TO_TARGET'
    wait_counter = 0

    total_len = 0.0
    prev_pos = P.pPos.Xc[[0, 1]].copy()
    folga_min = 1e9
    int_u2 = 0.0
    int_w2 = 0.0
    station_errs = []
    log = []

    while t < 300.0 and wp_idx < len(WAYPOINTS_ROTA):
        P.rGetSensorData()
        xc_true, yc_true = P.pPos.Xc[[0, 1]]
        psi_true = P.pPos.Xc[5]

        if ruido:
            xc = xc_true + rng.normal(0, 0.01)
            yc = yc_true + rng.normal(0, 0.01)
            psi = wrap_angle(psi_true + rng.normal(0, np.deg2rad(0.5)))
        else:
            xc, yc, psi = xc_true, yc_true, psi_true

        wp = WAYPOINTS_ROTA[wp_idx]
        name, tx, ty, tpsi_deg, is_stop = wp
        tpsi = np.deg2rad(tpsi_deg)

        dx = tx - xc
        dy = ty - yc
        rho = np.sqrt(dx * dx + dy * dy)
        target_heading = np.arctan2(dy, dx)
        err_heading = wrap_angle(target_heading - psi)
        u, w = 0.0, 0.0

        if phase == 'TURN_TO_TARGET':
            if abs(err_heading) < np.deg2rad(2.0):
                phase = 'DRIVE_STRAIGHT'
            else:
                w = np.clip(3.0 * err_heading, -W_MAX, W_MAX)

        elif phase == 'DRIVE_STRAIGHT':
            threshold = 0.025 if is_stop else 0.15
            if rho < threshold:
                if is_stop:
                    phase = 'TURN_TO_FINAL'
                else:
                    wp_idx += 1
                    phase = 'TURN_TO_TARGET'
                    continue
            else:
                u_max = 0.45
                u = max(0.05, u_max * (rho / 0.35)) if (is_stop and rho < 0.35) else u_max
                w = np.clip(4.0 * err_heading, -np.deg2rad(80), np.deg2rad(80))

        elif phase == 'TURN_TO_FINAL':
            err_final = wrap_angle(tpsi - psi)
            dist_true = np.linalg.norm(np.array([xc_true, yc_true]) - np.array([tx, ty]))
            if abs(err_final) < np.deg2rad(1.5):
                station_errs.append({
                    'name': name,
                    'rho_cm': dist_true * 100.0,
                    'psi_err_deg': np.rad2deg(abs(wrap_angle(tpsi - psi_true)))
                })
                phase = 'WAIT'
                wait_counter = 5
            else:
                w = np.clip(3.0 * err_final, -W_MAX, W_MAX)

        elif phase == 'WAIT':
            wait_counter -= 1
            if wait_counter <= 0:
                wp_idx += 1
                if wp_idx >= len(WAYPOINTS_ROTA):
                    break
                phase = 'TURN_TO_TARGET'

        u_sat, w_sat = saturate_ratio(u, w)
        if ruido:
            u_act = u_sat * (1.0 + rng.normal(0, 0.05))
            w_act = w_sat * (1.0 + rng.normal(0, 0.05))
            u_app, w_app = saturate_ratio(u_act, w_act)
        else:
            u_app, w_app = u_sat, w_sat

        P.pSC.Ud = np.array([u_app, w_app])
        P.rSendControlSignals(dt=dt)
        t += dt

        cur_pos = P.pPos.Xc[[0, 1]].copy()
        total_len += np.linalg.norm(cur_pos - prev_pos)
        prev_pos = cur_pos

        d_obs = dist_to_obstacles(xc_true, yc_true)
        if d_obs < folga_min:
            folga_min = d_obs
        int_u2 += (u_app ** 2) * dt
        int_w2 += (w_app ** 2) * dt

        log.append([t, xc_true, yc_true, psi_true, u_app, w_app])

    log_arr = np.array(log)
    dev_max_cm, dev_rms_cm = calcular_desvio_caminho_referencia(log_arr[:, 1:3], ref_pts)

    return {
        'nome': 'Duas Fases (Gira-Anda-Gira)',
        'tempo': t,
        'dist': total_len,
        'maior_desvio_cm': dev_max_cm,
        'rms_desvio_cm': dev_rms_cm,
        'min_obs': folga_min,
        'int_u2': int_u2,
        'int_w2': int_w2,
        'mean_rho': np.mean([s['rho_cm'] for s in station_errs]) if station_errs else 0.0,
        'mean_psi': np.mean([s['psi_err_deg'] for s in station_errs]) if station_errs else 0.0,
        'estacoes': station_errs,
        'log': log_arr
    }


def _imprimir_comparativo_d(subtitulo, res_b, res_polar, res_duas):
    """Imprime tabela comparativa das três estratégias para um dado cenário (com ou sem ruído)."""
    print("\n" + "=" * 86)
    print(f"TABELA COMPARATIVA DE DESEMPENHO — {subtitulo}")
    print("=" * 86)
    print(f"{'Métrica (Seção 5)':32s} | {'Seguidor Caminho':16s} | {'Lei Polar':16s} | {'Duas Fases':16s}")
    print("-" * 86)

    m_rho_b = np.mean([s['rho_cm'] for s in res_b['estacoes']])
    m_psi_b = np.mean([s['psi_err_deg'] for s in res_b['estacoes']])

    linhas = [
        ("1. Tempo total (s)", f"{res_b['tempo_total']:.1f}", f"{res_polar['tempo']:.1f}", f"{res_duas['tempo']:.1f}"),
        ("2. Comprimento percorrido (m)", f"{res_b['comprimento']:.2f}", f"{res_polar['dist']:.2f}", f"{res_duas['dist']:.2f}"),
        ("3. Erro estações rho (cm)", f"{m_rho_b:.2f}", f"{res_polar['mean_rho']:.2f}", f"{res_duas['mean_rho']:.2f}"),
        ("3. Erro estações |dpsi| (°)", f"{m_psi_b:.2f}", f"{res_polar['mean_psi']:.2f}", f"{res_duas['mean_psi']:.2f}"),
        ("4. Desvio da ref. Máx. (cm)", f"{res_b['maior_desvio_cm']:.2f}", f"{res_polar['maior_desvio_cm']:.2f}", f"{res_duas['maior_desvio_cm']:.2f}"),
        ("4. Desvio da ref. RMS (cm)", f"{res_b['rms_desvio_cm']:.2f}", f"{res_polar['rms_desvio_cm']:.2f}", f"{res_duas['rms_desvio_cm']:.2f}"),
        ("5. Folga mín. obstáculos (m)", f"{res_b['folga_minima']:.3f}", f"{res_polar['min_obs']:.3f} (VIOLA!)", f"{res_duas['min_obs']:.3f}"),
        ("6. Esforço linear int(u²) (m²/s)", f"{res_b['esforco_u2']:.2f}", f"{res_polar['int_u2']:.2f}", f"{res_duas['int_u2']:.2f}"),
        ("6. Esforço angular int(w²) (rad²/s)", f"{res_b['esforco_w2']:.2f}", f"{res_polar['int_w2']:.2f}", f"{res_duas['int_w2']:.2f}")
    ]

    for m_nome, v1, v2, v3 in linhas:
        print(f"{m_nome:32s} | {v1:16s} | {v2:16s} | {v3:16s}")
    print("=" * 86)


def executar_tarefa_d(salvar=SALVAR_GRAFICOS):
    """
    Tarefa D — Comparação das três estratégias (Seguidor de Caminho, Lei Polar, Duas Fases)
    COM e SEM ruído, conforme exigido na Seção 2 do enunciado.
    """
    print("\n" + "=" * 86)
    print("EXECUTANDO TAREFA D — COMPARAÇÃO DE ESTRATÉGIAS (SEM RUÍDO E COM RUÍDO)")
    print("=" * 86)

    print("\n[Cenário 1/2: SEM RUÍDO]")
    res_b_clean = executar_seguidor_caminho(ruido=False, animar=False, plotar=False)
    res_polar_clean = executar_estrategia_polar(ruido=False)
    res_duas_clean = executar_estrategia_duas_fases(ruido=False)
    _imprimir_comparativo_d("SEM RUÍDO", res_b_clean, res_polar_clean, res_duas_clean)

    print("\n[Cenário 2/2: COM RUÍDO DE ODOMETRIA E ATUAÇÃO (Semente = 42)]")
    res_b_noise = executar_seguidor_caminho(ruido=True, semente=42, animar=False, plotar=False)
    res_polar_noise = executar_estrategia_polar(ruido=True, semente=42)
    res_duas_noise = executar_estrategia_duas_fases(ruido=True, semente=42)
    _imprimir_comparativo_d("COM RUÍDO (Semente = 42)", res_b_noise, res_polar_noise, res_duas_noise)

    if salvar:
        os.makedirs(PASTA_FIGURAS, exist_ok=True)
        # Plot comparativo de trajetórias no mesmo cenário
        fig, ax = plt.subplots(figsize=(10, 9))
        cenario_linha_montagem(ax)
        ref_pts = obter_pontos_caminho_referencia(res_b_clean['segments'])
        ax.plot(ref_pts[:, 0], ref_pts[:, 1], 'k--', linewidth=1.2, alpha=0.6, label="Referência Oficial")
        ax.plot(res_b_clean['log'][:, 1], res_b_clean['log'][:, 2], 'g-', linewidth=2.2, label="1) Seguidor de Caminho (Recomendado)")
        ax.plot(res_duas_clean['log'][:, 1], res_duas_clean['log'][:, 2], 'b-.', linewidth=1.6, label="2) Duas Fases (Gira-Anda-Gira)")
        ax.plot(res_polar_clean['log'][:, 1], res_polar_clean['log'][:, 2], 'r:', linewidth=2.0, label="3) Lei Polar (Ondulações)")

        ax.set_title("Tarefa D: Comparação de Trajetórias das Três Estratégias", fontsize=12, fontweight='bold')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        fig.savefig(os.path.join(PASTA_FIGURAS, "comparacao_trajetorias_tarefa_D.png"), dpi=300, bbox_inches='tight')

        # Gráfico de barras comparativo de métricas principais (Sem Ruído vs Com Ruído)
        fig_bar, axs_bar = plt.subplots(1, 3, figsize=(13, 4.5))
        estrategias = ['Seguidor', 'Lei Polar', 'Duas Fases']
        x_pos = np.arange(len(estrategias))
        width = 0.35

        m_rho_b_clean = np.mean([s['rho_cm'] for s in res_b_clean['estacoes']])
        m_rho_b_noise = np.mean([s['rho_cm'] for s in res_b_noise['estacoes']])

        # 1. Erro de posição nas estações (Sem vs Com Ruído)
        axs_bar[0].bar(x_pos - width/2, [m_rho_b_clean, res_polar_clean['mean_rho'], res_duas_clean['mean_rho']],
                       width, label='Sem Ruído', color='#2ca02c')
        axs_bar[0].bar(x_pos + width/2, [m_rho_b_noise, res_polar_noise['mean_rho'], res_duas_noise['mean_rho']],
                       width, label='Com Ruído', color='#ff7f0e')
        axs_bar[0].axhline(3.0, color='r', linestyle='--', label='Tolerância (3.0 cm)')
        axs_bar[0].set_xticks(x_pos)
        axs_bar[0].set_xticklabels(estrategias)
        axs_bar[0].set_ylabel("Erro $\\rho$ (cm)")
        axs_bar[0].set_title("Erro Médio nas Estações")
        axs_bar[0].grid(True, axis='y', linestyle=':')
        axs_bar[0].legend(fontsize=8)

        # 2. Folga mínima a obstáculos
        axs_bar[1].bar(x_pos - width/2, [res_b_clean['folga_minima'], res_polar_clean['min_obs'], res_duas_clean['min_obs']],
                       width, label='Sem Ruído', color='#1f77b4')
        axs_bar[1].bar(x_pos + width/2, [res_b_noise['folga_minima'], res_polar_noise['min_obs'], res_duas_noise['min_obs']],
                       width, label='Com Ruído', color='#9467bd')
        axs_bar[1].axhline(0.35, color='r', linestyle='--', label="Limite Mínimo (0.35 m)")
        axs_bar[1].set_xticks(x_pos)
        axs_bar[1].set_xticklabels(estrategias)
        axs_bar[1].set_ylabel("Distância (m)")
        axs_bar[1].set_title("Folga Mínima a Obstáculos")
        axs_bar[1].grid(True, axis='y', linestyle=':')
        axs_bar[1].legend(fontsize=8, loc='lower right')

        # 3. Esforço angular
        axs_bar[2].bar(x_pos - width/2, [res_b_clean['esforco_w2'], res_polar_clean['int_w2'], res_duas_clean['int_w2']],
                       width, label='Sem Ruído', color='#2ca02c')
        axs_bar[2].bar(x_pos + width/2, [res_b_noise['esforco_w2'], res_polar_noise['int_w2'], res_duas_noise['int_w2']],
                       width, label='Com Ruído', color='#d62728')
        axs_bar[2].set_xticks(x_pos)
        axs_bar[2].set_xticklabels(estrategias)
        axs_bar[2].set_ylabel("rad²/s")
        axs_bar[2].set_title("Esforço Angular $\\int \\omega^2 dt$")
        axs_bar[2].grid(True, axis='y', linestyle=':')
        axs_bar[2].legend(fontsize=8)

        fig_bar.tight_layout()
        fig_bar.savefig(os.path.join(PASTA_FIGURAS, "comparacao_metricas_tarefa_D.png"), dpi=300, bbox_inches='tight')
        plt.show(block=False)

    return (res_b_clean, res_polar_clean, res_duas_clean), (res_b_noise, res_polar_noise, res_duas_noise)


# ==============================================================================
# BLOCO PRINCIPAL
# ==============================================================================
def main():
    global TAREFA, HABILITAR_RUIDO, ANIMACAO

    # Leitura de argumentos por linha de comando (se fornecidos)
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper().replace('-', '')
        if arg in ['A', 'B', 'C', 'D', 'TODAS']:
            TAREFA = arg
        if '--ruido' in sys.argv:
            HABILITAR_RUIDO = True
        if '--animar' in sys.argv:
            ANIMACAO = True

    print("\n" + "#" * 75)
    print("ROBÓTICA TERRESTRE — IFBA CAMPUS VITÓRIA DA CONQUISTA")
    print(f"Executando modo: TAREFA = '{TAREFA}', RUÍDO = {HABILITAR_RUIDO}, ANIMAÇÃO = {ANIMACAO}")
    print("#" * 75)

    if TAREFA == 'A':
        executar_tarefa_a()
    elif TAREFA == 'B':
        res = executar_seguidor_caminho(ruido=HABILITAR_RUIDO, animar=ANIMACAO, plotar=True)
        m_rho = float(np.mean([s['rho_cm'] for s in res['estacoes']]))
        m_psi = float(np.mean([s['psi_err_deg'] for s in res['estacoes']]))
        imprimir_tabela_metricas(
            f"TABELA OFICIAL DE MÉTRICAS — TAREFA B (RUÍDO = {HABILITAR_RUIDO})",
            res['tempo_total'], res['comprimento'], m_rho, m_psi,
            res['maior_desvio_cm'], res['rms_desvio_cm'], res['folga_minima'],
            res['esforco_u2'], res['esforco_w2']
        )
        print("\nDetalhamento de Parada nas Estações (Requisito: rho < 3.0 cm e |dpsi| < 2.0°):")
        for st in res['estacoes']:
            ok = "APROVADO" if (st['rho_cm'] < 3.0 and st['psi_err_deg'] < 2.0) else "REPROVADO"
            print(f"  - {st['name']:8s}: rho = {st['rho_cm']:.2f} cm | |dpsi| = {st['psi_err_deg']:.2f}° | [{ok}]")
    elif TAREFA == 'C':
        executar_tarefa_c(num_execucoes=10)
    elif TAREFA == 'D':
        executar_tarefa_d()
    elif TAREFA == 'TODAS':
        executar_tarefa_a()
        res_b = executar_seguidor_caminho(ruido=False, animar=False, plotar=True)
        m_rho = float(np.mean([s['rho_cm'] for s in res_b['estacoes']]))
        m_psi = float(np.mean([s['psi_err_deg'] for s in res_b['estacoes']]))
        imprimir_tabela_metricas(
            "TABELA OFICIAL DE MÉTRICAS — TAREFA B (SEM RUÍDO)",
            res_b['tempo_total'], res_b['comprimento'], m_rho, m_psi,
            res_b['maior_desvio_cm'], res_b['rms_desvio_cm'], res_b['folga_minima'],
            res_b['esforco_u2'], res_b['esforco_w2']
        )
        executar_tarefa_c(num_execucoes=10)
        executar_tarefa_d()

    print("\nExecução finalizada com sucesso!")


if __name__ == '__main__':
    main()
