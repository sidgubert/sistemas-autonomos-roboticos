"""
Script para geração de animação/vídeo da simulação da Tarefa B (Roteiro Completo do Pioneer 3DX).

Pode ser executado de dois modos:
1) Modo Interativo (Tempo Real):
   python gerar_animacao_tarefa_B.py --tempo-real
   Abre a janela gráfica com o cenário e o robô Pioneer 3DX percorrendo o trajeto.

2) Modo Exportação (GIF Animado / Vídeo):
   python gerar_animacao_tarefa_B.py
   Gera um arquivo animado otimizado em 'figuras/simulacao_tarefa_B.gif' mostrando todo o percurso.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pioneer3dx import Pioneer3DX
from cenario_linha_montagem import cenario_linha_montagem
from avaliacao_p3dx import construir_caminho_referencia, obter_pontos_caminho_referencia, wrap_angle, saturate_ratio, DT_NOMINAL, W_MAX

PASTA_FIGURAS = 'figuras'
NOME_GIF = os.path.join(PASTA_FIGURAS, 'simulacao_tarefa_B.gif')


def gerar_animacao(modo_tempo_real=False, salvar_gif=True, fps=10, amostras_pulo=3):
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    segments = construir_caminho_referencia(raio_arco=0.5)

    P = Pioneer3DX()
    P.rSetPose(0.0, -5.0, np.deg2rad(90.0))

    dt = DT_NOMINAL
    t = 0.0
    seg_idx = 0
    state = 'ALIGN_START'
    wait_counter = 0

    u_nom = 0.42
    u_arc = 0.32
    k_psi = 5.0
    k_y = 10.0

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    cenario_linha_montagem(ax)
    ref_pts = obter_pontos_caminho_referencia(segments)
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], 'm--', linewidth=1.2, alpha=0.6, label="Referência Oficial")
    
    rastro_x = [P.pPos.Xc[0]]
    rastro_y = [P.pPos.Xc[1]]
    linha_rastro, = ax.plot(rastro_x, rastro_y, 'b-', linewidth=1.8, label="P3DX Trajetória")
    P.mCADplot(ax)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
    title_obj = ax.set_title("Simulação Tarefa B — Início", fontsize=11, fontweight='bold')

    frames = []
    step = 0

    if modo_tempo_real:
        plt.ion()
        plt.show()

    print("Simulando trajetória da Tarefa B...")
    while t < 300.0 and seg_idx < len(segments):
        P.rGetSensorData()
        xc, yc = P.pPos.Xc[[0, 1]]
        psi = P.pPos.Xc[5]
        cur_seg = segments[seg_idx]

        u = 0.0
        w = 0.0

        if state == 'ALIGN_START':
            target_psi = segments[0]['heading']
            err_psi = wrap_angle(target_psi - psi)
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

                v = np.array([xc, yc]) - p1
                s = np.dot(v, tangent)
                e_y = np.dot(v, normal)
                e_psi = wrap_angle(heading - psi)
                kappa = 0.0

                rem_dist = L - s
                if cur_seg['is_station'] and rem_dist <= 0.35:
                    dist_to_end = np.linalg.norm(np.array([xc, yc]) - p2)
                    u_ref = max(0.04, u_nom * (dist_to_end / 0.40))
                    if dist_to_end < 0.015 or rem_dist <= 0.005:
                        state = 'STOP_AT_STATION'
                        u, w = 0.0, 0.0
                    else:
                        u = u_ref * max(np.cos(e_psi), 0.0)
                        w = u * kappa + k_psi * e_psi - k_y * u * e_y
                elif not cur_seg['is_station'] and rem_dist <= 0.03:
                    seg_idx += 1
                    continue
                else:
                    u = u_nom * max(np.cos(e_psi), 0.0)
                    w = u * kappa + k_psi * e_psi - k_y * u * e_y

            elif cur_seg['type'] == 'arc':
                center = cur_seg['center']
                r = cur_seg['r']
                t1 = cur_seg['t1']
                ccw = cur_seg['ccw']
                kappa = cur_seg['kappa']
                L = cur_seg['length']

                v = np.array([xc, yc]) - center
                dist_center = np.linalg.norm(v)
                angle_cur = np.arctan2(v[1], v[0])

                if ccw:
                    heading = angle_cur + np.pi / 2.0
                    e_y = r - dist_center
                    d_theta = (angle_cur - t1) % (2.0 * np.pi)
                    arc_progress = 0.0 if d_theta > np.pi else r * d_theta
                else:
                    heading = angle_cur - np.pi / 2.0
                    e_y = dist_center - r
                    d_theta = (t1 - angle_cur) % (2.0 * np.pi)
                    arc_progress = 0.0 if d_theta > np.pi else r * d_theta

                e_psi = wrap_angle(heading - psi)
                rem_dist = L - arc_progress
                if rem_dist <= 0.03:
                    seg_idx += 1
                    continue
                else:
                    u = u_arc * max(np.cos(e_psi), 0.0)
                    w = u * kappa + k_psi * e_psi - k_y * u * e_y

        elif state == 'STOP_AT_STATION':
            psi_f = cur_seg['psi_f']
            err_psi = wrap_angle(psi_f - psi)
            if abs(err_psi) < np.deg2rad(1.0):
                state = 'WAIT_STATION'
                wait_counter = 4
                u, w = 0.0, 0.0
            else:
                u = 0.0
                w = np.clip(3.0 * err_psi, -np.deg2rad(80), np.deg2rad(80))

        elif state == 'WAIT_STATION':
            u, w = 0.0, 0.0
            wait_counter -= 1
            if wait_counter <= 0:
                if seg_idx == len(segments) - 1:
                    break
                seg_idx += 1
                state = 'REALIGN_PATH'

        elif state == 'REALIGN_PATH':
            next_heading = segments[seg_idx]['heading']
            err_psi = wrap_angle(next_heading - psi)
            if abs(err_psi) < np.deg2rad(1.0):
                state = 'FOLLOW'
            else:
                u = 0.0
                w = np.clip(3.0 * err_psi, -np.deg2rad(80), np.deg2rad(80))

        u_sat, w_sat = saturate_ratio(u, w)
        P.pSC.Ud = np.array([u_sat, w_sat])
        P.rSendControlSignals(dt=dt)
        t += dt
        step += 1

        rastro_x.append(xc)
        rastro_y.append(yc)

        # Captura de frames periodicamente para o GIF animado
        if step % amostras_pulo == 0:
            P.mCADplot(ax)
            linha_rastro.set_data(rastro_x, rastro_y)
            seg_nome = cur_seg.get('name', '')
            title_obj.set_text(f"P3DX Linha de Montagem | Tempo: {t:05.1f}s | Trecho: {seg_nome}")

            if modo_tempo_real:
                plt.pause(0.001)

            if salvar_gif:
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                im = Image.fromarray(rgba).convert('P', palette=Image.Palette.ADAPTIVE)
                frames.append(im)

    if modo_tempo_real:
        plt.ioff()

    if salvar_gif and frames:
        print(f"Salvando animação com {len(frames)} quadros em '{NOME_GIF}'...")
        # Salva o GIF animado com loop contínuo
        frames[0].save(
            NOME_GIF,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True
        )
        print(f"Arquivo gerado com sucesso: {NOME_GIF} ({os.path.getsize(NOME_GIF) / 1024:.1f} KB)")

    plt.close(fig)


if __name__ == '__main__':
    tempo_real = '--tempo-real' in sys.argv
    gerar_animacao(modo_tempo_real=tempo_real, salvar_gif=True, fps=12, amostras_pulo=4)
