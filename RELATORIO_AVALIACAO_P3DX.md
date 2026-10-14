# INSTITUTO FEDERAL DA BAHIA — CAMPUS VITÓRIA DA CONQUISTA
## Disciplina: Robótica Terrestre
### Avaliação — Navegação Contínua do Pioneer 3DX na Linha de Montagem

---

**Equipe:** Engenharia / Sistemas Autônomos Robóticos  
**Plataforma de Simulação:** Pioneer 3DX (AuRoRA / Python 3 + NumPy + Matplotlib)  
**Data:** Outubro de 2026  

---

## 1. Resumo Executivo e Objetivos

Nesta avaliação, foi desenvolvido e validado um sistema completo de controle e navegação contínua para o robô móvel diferencial **Pioneer 3DX** operando em uma planta industrial de linha de montagem com 4 blocos fabris e 8 estações de atendimento.

O projeto contemplou quatro tarefas essenciais:
1. **Tarefa A (Controle Contínuo até uma Pose):** Implementação e sintonia da lei de controle não-linear em coordenadas polares (Siegwart / Aicardi) com comprovação de estabilidade de Lyapunov e tratamento de alvos retrógrados ($|\alpha| > 90^\circ$).
2. **Tarefa B (Seguidor de Caminho Suave):** Síntese de um seguidor de caminho com antecipação de curvatura (feedforward $\kappa$) e realimentação de erro lateral ($e_y$) e erro de rumo ($e_\psi$) sobre um circuito composto por retas e arcos de raio $R = 0{,}5\text{ m}$ tangentes nas quinas C1 a C8. O robô contorna as quinas sem parar e para nas 8 estações e na Base dentro das tolerâncias estritas ($\rho < 3\text{ cm}$, $|\Delta\psi| < 2^\circ$, desvio lateral $\le 5\text{ cm}$).
3. **Tarefa C (Robustez a Ruído):** Avaliação estatística em 10 execuções independentes sob ruído de odometria gaussiano ($\sigma_{x,y} = 1\text{ cm}$, $\sigma_\psi = 0{,}5^\circ$) e ruído de atuação multiplicativo ($\mathcal{N}(0, 0{,}05)$).
4. **Tarefa D (Análise Comparativa de Três Estratégias):** Comparativo experimental quantitativo entre o **Seguidor de Caminho**, a **Lei Polar Ponto-a-Ponto** e o **Controlador em Duas Fases (Gira-Anda-Gira)** pelas 6 métricas de desempenho normalizadas da disciplina.

---

## 2. Modelagem Cinemática e Leis de Controle

### 2.1 Modelo Cinemático do Pioneer 3DX
O Pioneer 3DX é modelado como um robô diferencial com restrição não-holonômica do tipo uniciclo, com centro de coordenadas no centro do eixo das rodas ($X_c = [x_c, y_c, \psi]^T$):

$$\begin{bmatrix} \dot{x}_c \\ \dot{y}_c \\ \dot{\psi} \end{bmatrix} = \begin{bmatrix} \cos\psi & 0 \\ \sin\psi & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} u \\ \omega \end{bmatrix}$$

Limites físicos de saturação:
- Velocidade linear: $|u| \le 0{,}75\text{ m/s}$
- Velocidade angular: $|\omega| \le 100^\circ/\text{s} \approx 1{,}7453\text{ rad/s}$

Para preservar o raio de curvatura instantâneo $R = u/\omega$ e a geometria do movimento quando qualquer canal atinge o limite físico, foi implementada a **saturação proporcional preservando a razão $u/\omega$**:

$$s = \max\left(1{,}0,\, \frac{|u|}{u_{\max}},\, \frac{|\omega|}{\omega_{\max}}\right), \quad u_{\text{sat}} = \frac{u}{s}, \quad \omega_{\text{sat}} = \frac{\omega}{s}$$

---

### 2.2 Tarefa A — Lei de Controle em Coordenadas Polares
Para conduzir o robô até uma pose desejada $(x_d, y_d, \psi_f)$ partindo de $(x, y, \psi)$, define-se a transformação de coordenadas polares no referencial local da linha de visada:

$$\rho = \sqrt{(x_d - x)^2 + (y_d - y)^2}$$
$$\alpha = \text{atan2}(y_d - y,\, x_d - x) - \psi$$
$$\beta = \psi_f - \psi - \alpha = \psi_f - \text{atan2}(y_d - y,\, x_d - x)$$

com $\alpha, \beta \in [-\pi, \pi]$. A lei de controle cinemático é dada por:

$$u = k_\rho \rho \max(\cos\alpha, 0)$$
$$\omega = k_\alpha \alpha + k_\beta \beta$$

#### Condições de Estabilidade de Lyapunov:
Pelo teorema de Aicardi et al. e Siegwart, a função candidata de Lyapunov $V = \frac{1}{2}\rho^2 + \frac{1}{2}\alpha^2 + \frac{|k_\beta|}{2 k_\alpha}\beta^2$ garante estabilidade assintótica local sse:

$$k_\rho > 0, \quad k_\beta < 0, \quad k_\alpha - k_\rho > 0$$

**Ganhos adotados e validados:**
- $k_\rho = 0{,}50\text{ s}^{-1}$
- $k_\alpha = 1{,}50\text{ s}^{-1}$ (satisfaz $k_\alpha - k_\rho = 1{,}00 > 0$)
- $k_\beta = -0{,}30\text{ s}^{-1}$ (satisfaz $k_\beta < 0$)

#### Tratamento de Alvos Retrógrados ($|\alpha| > 90^\circ$):
Quando o alvo encontra-se atrás do robô ($|\alpha| > \pi/2$), a projeção $\cos\alpha < 0$. Se a lei linear direta $u = k_\rho \rho$ fosse mantida sem modulação, o robô aceleraria para a frente enquanto tentava virar, produzindo uma laçada frontal ampla e perigosa (*cusp*). A equipe adotou a modulação suave $u = k_\rho \rho \max(0, \cos\alpha)$, que zera suavemente a translação para alvos posteriores, priorizando a rotação no lugar até que o robô aponte para o quadrante do alvo ($|\alpha| \le 90^\circ$).

---

### 2.3 Tarefa B — Seguidor de Caminho Contínuo (Retas e Arcos)
O caminho de referência $\Gamma(s)$ foi parametrizado rigorosamente conforme a Seção 3 do enunciado:
- Comprimento total da rota: **$57{,}08\text{ m}$** (conforme os $57{,}1\text{ m}$ previstos pelo cálculo analítico).
- Quinas C1 a C8: arcos circulares de raio $R = 0{,}5\text{ m}$ tangentes às retas de entrada e saída.
- Curvatura nominal: $\kappa = 0$ nas retas; $\kappa = +1/R = +2{,}0\text{ m}^{-1}$ em curvas anti-horárias (CCW: C1, C2, C3, C6, C7, C8); $\kappa = -1/R = -2{,}0\text{ m}^{-1}$ em curvas horárias (CW: C4, C5).

No referencial local de Serret-Frenet do ponto mais próximo da trajetória, o erro de rumo $e_\psi$ e o erro lateral $e_y$ são dados por:
$$e_\psi = \text{atan2}(\sin(\psi_{\text{caminho}} - \psi),\, \cos(\psi_{\text{caminho}} - \psi))$$
$$e_y = -(x - x_r)\sin\psi_r + (y - y_r)\cos\psi_r$$

A lei de controle do seguidor de caminho é expressa por:

$$u = u_{\text{ref}} \max(\cos e_\psi,\, 0)$$
$$\omega = u \kappa + k_\psi e_\psi - k_y u e_y$$

onde:
- $u \kappa$: componente de antecipação (*feedforward*) de curvatura, que comanda a rotação necessária para percorrer o arco sem atraso de fase.
- $k_\psi e_\psi$: realimentação proporcional de erro de orientação.
- $-k_y u e_y$: realimentação lateral proporcional à velocidade linear $u$.

**Ganhos sintonizados da Tarefa B:**
- Velocidade de cruzeiro nas retas: $u_{\text{nom}} = 0{,}42\text{ m/s}$
- Velocidade segura em curvas: $u_{\text{arc}} = 0{,}32\text{ m/s}$
- Ganho angular: $k_\psi = 5{,}0\text{ s}^{-1}$
- Ganho lateral: $k_y = 10{,}0\text{ m}^{-2}$

---

## 3. Resultados Experimentais e Validação das Tarefas

### 3.1 Resultados da Tarefa A (Três Casos de Pose)

| Caso de Teste | Pose Alvo $(x_d, y_d, \psi_f)$ | Tempo (s) | Erro Posição $\rho$ (cm) | Erro Ângulo $\|\Delta\psi\|$ (°) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1) Destino à frente** | $(3{,}0\text{ m},\, 1{,}0\text{ m},\, 0^\circ)$ | $10{,}6$ | $2{,}89\text{ cm}$ | $1{,}44^\circ$ | **CONVERGIU** |
| **2) Destino ao lado** | $(0{,}0\text{ m},\, 2{,}0\text{ m},\, 90^\circ)$ | $9{,}6$ | $2{,}87\text{ cm}$ | $1{,}62^\circ$ | **CONVERGIU** |
| **3) Destino atrás** | $(-2{,}0\text{ m},\, 0{,}0\text{ m},\, 180^\circ)$ | $10{,}5$ | $2{,}87\text{ cm}$ | $1{,}62^\circ$ | **CONVERGIU** |

*Critério de aceitação:* $\rho < 3{,}0\text{ cm}$ e $|\Delta\psi| < 2{,}0^\circ$ — **100% cumprido**.

---

### 3.2 Resultados da Tarefa B (Roteiro Completo Sem Ruído)

- **Tempo Total:** $196{,}7\text{ s}$ (Requisito: $\le 300\text{ s}$) $\to$ Margem de segurança temporal de 34,4%.
- **Comprimento Percorrido:** $57{,}21\text{ m}$ (Nominal: $57{,}1\text{ m}$) $\to$ Rastreamento estrito sem desvios parasitas.
- **Maior Desvio da Rota:** **$2{,}06\text{ cm}$** (Requisito: $\le 5{,}0\text{ cm}$) $\to$ Precisão mais de duas vezes superior à exigência da disciplina!
- **Desvio RMS da Rota:** **$0{,}68\text{ cm}$**.
- **Folga Mínima aos Obstáculos:** **$0{,}579\text{ m}$** (Requisito: $\ge 0{,}35\text{ m}$) $\to$ Margem de segurança de $0{,}229\text{ m}$ acima do limite de colisão.
- **Esforço de Controle:** $\int u^2 dt = 22{,}73\text{ m}^2/\text{s}$, $\int \omega^2 dt = 42{,}77\text{ rad}^2/\text{s}$.

#### Desempenho nas Paradas de Atendimento (8 Estações + Base):

| Estação | Tipo | Localização $(x, y)$ | $\psi_f$ Desejado | Erro $\rho$ (cm) | Erro Ângulo (°) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEe** | Externa | $(5{,}0,\, -2{,}5)$ | $180^\circ$ | $1{,}30\text{ cm}$ | $0{,}73^\circ$ | Aprovado |
| **NEe** | Externa | $(5{,}0,\, 2{,}5)$ | $180^\circ$ | $1{,}29\text{ cm}$ | $0{,}73^\circ$ | Aprovado |
| **NEi** | Interna | $(1{,}4,\, 2{,}5)$ | $0^\circ$ | $1{,}43\text{ cm}$ | $0{,}73^\circ$ | Aprovado |
| **SEi** | Interna | $(1{,}4,\, -2{,}5)$ | $0^\circ$ | $1{,}30\text{ cm}$ | $0{,}73^\circ$ | Aprovado |
| **SOi** | Interna | $(-1{,}4,\, -2{,}5)$ | $180^\circ$ | $1{,}23\text{ cm}$ | $0{,}74^\circ$ | Aprovado |
| **NOi** | Interna | $(-1{,}4,\, 2{,}5)$ | $180^\circ$ | $1{,}26\text{ cm}$ | $0{,}73^\circ$ | Aprovado |
| **NOe** | Externa | $(-5{,}0,\, 2{,}5)$ | $0^\circ$ | $1{,}43\text{ cm}$ | $0{,}73^\circ$ | Aprovado |
| **SOe** | Externa | $(-5{,}0,\, -2{,}5)$ | $0^\circ$ | $1{,}30\text{ cm}$ | $0{,}73^\circ$ | Aprovado |
| **Base** | Chegada | $(0{,}0,\, -5{,}0)$ | $90^\circ$ | $1{,}34\text{ cm}$ | $0{,}73^\circ$ | Aprovado |

---

### 3.3 Resultados da Tarefa C (Robustez a Ruído — 10 Execuções)

Sob perturbações simultâneas de odometria estocástica e imperfeição de torque nos atuadores, os resultados consolidados foram:

| Métrica Oficial (Seção 5) | Valor Médio $\pm$ Desvio Padrão | Unidade | Valor Exigido | Situação |
| :--- | :---: | :---: | :---: | :---: |
| **1. Tempo total** | **$194{,}500 \pm 0{,}645$** | $\text{s}$ | $\le 300\text{ s}$ | **Atende** |
| **2. Comprimento percorrido** | **$57{,}207 \pm 0{,}019$** | $\text{m}$ | Ref. $\approx 57{,}1\text{ m}$ | **Atende** |
| **3. Erro de pose (distância $\rho$)** | **$1{,}648 \pm 0{,}258$** | $\text{cm}$ | $< 3{,}0\text{ cm}$ | **Atende** |
| **3. Erro de pose (orientação $|\Delta\psi|$)** | **$0{,}867 \pm 0{,}078$** | $^\circ$ | $< 2{,}0^\circ$ | **Atende** |
| **4. Desvio da referência (Máximo)** | **$2{,}474 \pm 0{,}227$** | $\text{cm}$ | $\le 5{,}0\text{ cm}$ | **Atende** |
| **4. Desvio da referência (RMS)** | **$0{,}748 \pm 0{,}046$** | $\text{cm}$ | — | Excelente |
| **5. Folga mínima a obstáculos** | **$0{,}579 \pm 0{,}003$** | $\text{m}$ | $\ge 0{,}35\text{ m}$ | **Atende** |
| **6. Esforço linear ($\int u^2 dt$)** | **$22{,}785 \pm 0{,}025$** | $\text{m}^2/\text{s}$ | — | Otimizado |
| **6. Esforço angular ($\int \omega^2 dt$)** | **$43{,}638 \pm 0{,}162$** | $\text{rad}^2/\text{s}$ | — | Otimizado |

---

### 3.4 Resultados da Tarefa D (Comparação de Estratégias)

| Métrica de Desempenho | Seguidor de Caminho (Tarefa B) | Lei Polar Ponto a Ponto | Duas Fases (Gira-Anda-Gira) |
| :--- | :---: | :---: | :---: |
| **Tempo Total (s)** | $196{,}7\text{ s}$ | $209{,}7\text{ s}$ | $\mathbf{184{,}1\text{ s}}$ |
| **Comprimento Percorrido (m)** | $\mathbf{57{,}21\text{ m}}$ | $66{,}36\text{ m}$ (+16%) | $57{,}69\text{ m}$ |
| **Erro médio $\rho$ estações (cm)** | $\mathbf{1{,}31\text{ cm}}$ | $2{,}93\text{ cm}$ | $2{,}23\text{ cm}$ |
| **Erro médio $|\Delta\psi|$ estações (°)** | $\mathbf{0{,}73^\circ}$ | $1{,}13^\circ$ | $1{,}29^\circ$ |
| **Maior desvio da rota (cm)** | $\mathbf{2{,}06\text{ cm}}$ | Incompatível (ondulações) | Incompatível (quinas vivas) |
| **Folga Mínima a Obstáculos (m)** | $\mathbf{0{,}579\text{ m}}$ | **$0{,}009\text{ m}$ (Quase Colisão!)** | $0{,}519\text{ m}$ |
| **Violação de Segurança ($\ge 0{,}35\text{ m}$)** | **NENHUMA** | **CRÍTICA (Violada!)** | NENHUMA |
| **Esforço Linear $\int u^2 dt$** | $\mathbf{22{,}73\text{ m}^2/\text{s}}$ | $37{,}02\text{ m}^2/\text{s}$ | $25{,}39\text{ m}^2/\text{s}$ |
| **Esforço Angular $\int \omega^2 dt$** | **$42{,}77\text{ rad}^2/\text{s}$** | $25{,}11\text{ rad}^2/\text{s}$ | $59{,}53\text{ rad}^2/\text{s}$ (+39%) |

---

## 4. Respostas Aprofundadas às Questões Orientadoras

### Questão 1: Como os ganhos $k_\psi$ e $k_y$ do seguidor de caminho afetam o desvio em relação às retas? O que acontece sem a correção lateral ($k_y = 0$) e com $k_y$ muito alto?
**Resposta:**
O termo angular $\omega = u\kappa + k_\psi e_\psi - k_y u e_y$ atua em duas variáveis de estado acopladas da cinemática de Serret-Frenet: a orientação do robô ($\dot{e}_\psi \approx -\omega + u\kappa$) e o deslocamento lateral ($\dot{e}_y = u \sin e_\psi \approx u e_\psi$).
1. **Sem correção lateral ($k_y = 0$):** O laço fecha apenas sobre a orientação ($e_\psi \to 0$). Se o robô possuir qualquer erro de posição inicial ou sofrer distúrbio transitório, ele alinhará sua orientação paralelamente à reta de referência, mas navegará com um deslocamento perpendicular constante e não-nulo ($e_y \neq 0$). Em outras palavras, sem realimentação de $e_y$, o sistema possui um integrador aberto na posição transversal, sendo incapaz de anular erros de rastreamento lateral.
2. **Com $k_y$ excessivamente alto:** O polo do sistema em malha fechada é empurrado para frequências elevadas. Para pequenos erros laterais, o ganho comanda variações abruptas de $\omega$, levando à saturação frequente de velocidade angular ($|\omega| = 100^\circ/\text{s}$). Como a frequência de controle é de $10\text{ Hz}$ ($\Delta t = 0{,}1\text{ s}$), essa hipercorreção produz atraso de fase, originando sobre-elevação (*overshoot*) e oscilações em zigue-zague ao redor da reta (*hunting effect*), aumentando o erro RMS e degradando a vida útil dos atuadores.
3. **Sintonia ótima:** A combinação $k_\psi = 5{,}0$ e $k_y = 10{,}0$ estabeleceu um fator de amortecimento crítico ($\zeta \approx 0{,}7$ a $0{,}8$), assegurando convergência monotônica rápida sem ondulações e mantendo o desvio máximo em extraordinários $2{,}06\text{ cm}$.

---

### Questão 2: A saturação de $u$ e $\omega$ altera a garantia de estabilidade? Por que, no seguidor de caminho, convém limitar $u$ antes de calcular $\omega$?
**Resposta:**
1. **Efeito da saturação na estabilidade:** Sim. As provas formais de estabilidade de Lyapunov assumem que os atuadores possuem autoridade linear irrestrita. Quando ocorre saturação desregulada (componente a componente via *clipping* independente), a razão cinemática $u/\omega$ é corrompida. Se $\omega$ saturar e $u$ não, o raio de giro instantâneo $R = u/\omega$ cresce indevidamente, fazendo o robô abrir a curva e afastar-se do caminho de referência, o que pode provocar perda de estabilidade ou colisão. A técnica adotada pela equipe de **saturar $u$ e $\omega$ em bloco pelo mesmo fator de escala $s$** preserva rigorosamente a geometria da trajetória, transformando a saturação em uma mera desaceleração temporal do avanço, preservando a estabilidade e a convergência.
2. **Por que limitar $u$ antes de calcular $\omega$:** Na lei de controle do seguidor de caminho, os termos dinâmicos dependem diretamente de $u$:
   - O termo feedforward de curvatura é $u \kappa$.
   - O termo de realimentação lateral é $-k_y u e_y$.
   Se $\omega$ for calculado utilizando uma velocidade de referência teórica elevada $u_{\text{nom}}$, mas depois o atuador linear saturar para um $u$ real menor, haverá um descasamento severo (*oversteering*): o robô estará girando na velocidade angular correspondente a uma velocidade de translação que ele não tem. Portanto, calcular $u$ (respeitando saturações, curvaturas e aproximações de estação) e inseri-lo no cálculo de $\omega$ garante rigorosa consistência cinemática instantânea.

---

### Questão 3: Por que a lei polar não é definida em $\rho = 0$? Como a equipe evitou oscilações perto das estações?
**Resposta:**
1. **Singularidade em $\rho = 0$:** As coordenadas polares são dadas por $\alpha = \text{atan2}(\Delta y, \Delta x) - \psi$ e $\beta = \psi_f - \psi - \alpha$. Quando o ponto de controle atinge com precisão o alvo $(\Delta x = 0, \Delta y = 0)$, tem-se $\rho = 0$. Nesse ponto, a função $\text{atan2}(0, 0)$ é matematicamente indeterminada, pois não existe um vetor diretor da linha de visada.
2. **Origem das oscilações terminais:** Conforme $\rho \to 0$, variações submilimétricas de posição causadas por ruído de medição ou truncamento numérico fazem o vetor $[\Delta x, \Delta y]$ girar arbitrariamente entre quadrantes opostos, variando $\text{atan2}$ em até $\pm 180^\circ$. Isso provocaria comandos instantâneos de $\omega$ oscilando entre $+100^\circ/\text{s}$ e $-100^\circ/\text{s}$.
3. **Solução implementada:** A equipe implementou uma máquina de estados com chaveamento de histerese em torno do raio de chegada: ao entrar na vizinhança $\rho < 3\text{ cm}$, a translação é desativada ($u = 0$), desacoplando a posição da orientação. Em seguida, aciona-se um controlador puramente angular proporcional $w = k_w \text{wrap}(\psi_f - \psi)$ com zona morta em $|\psi_f - \psi| < 1{,}5^\circ$, garantindo chegada suave, convergência estrita e parada com velocidade nula sem vibrações.

---

### Questão 4: Qual é o efeito do raio das curvas nas quinas sobre o tempo total, o desvio do caminho e a velocidade angular? Que limite físico do P3DX restringe o raio mínimo?
**Resposta:**
1. **Efeito no tempo total e velocidade linear:** Em um arco de raio $R$, a velocidade angular necessária é $\omega = u / R$. Se o raio for muito pequeno, para não estourar o limite de saturação $\omega_{\max}$, a velocidade linear deve ser drasticamente reduzida ($u \le R \omega_{\max}$). Arcos com $R = 0{,}5\text{ m}$ permitem velocidades de contorno de até $0{,}4\text{ m/s}$ mantendo $\omega \approx 45^\circ/\text{s}$, sem necessidade de parar nas quinas, reduzindo significativamente o tempo de percurso em relação a quinas vivas de $90^\circ$.
2. **Efeito no desvio do caminho:** A inserção de curvas tangentes confere continuidade $G^1$ (tangente contínua) à trajetória. Como o Pioneer possui atuadores reais com inércia e frequência finita de controle (10 Hz), o robô é fisicamente incapaz de fazer uma quina ortogonal a $90^\circ$ sem parar. O arco de $0{,}5\text{ m}$ elimina descontinuidades angulares bruscas, permitindo manter o desvio do caminho abaixo de $2{,}1\text{ cm}$.
3. **Limite físico restritivo:** O fator determinante no Pioneer 3DX é a saturação da velocidade angular máxima ($\omega_{\max} = 100^\circ/\text{s} = 1{,}7453\text{ rad/s}$) e a largura de bitola das rodas ($L_{\text{bitola}} \approx 0{,}33\text{ m}$). Para contornar uma curva a uma velocidade mínima aceitável sem parar ($u > 0$), o raio mínimo cinemático é:
   $$R_{\min} = \frac{u}{\omega_{\max}}$$
   Se $u = 0{,}35\text{ m/s}$, tem-se $R_{\min} \approx 0{,}20\text{ m}$. Abaixo disso, as rodas internas e externas sofrem inversão de rotação relativa severa, forçando o robô a girar no próprio eixo (estratégia de parada total). O raio de $0{,}5\text{ m}$ adotado no enunciado situa-se em uma faixa ideal de estabilidade e eficiência dinâmica.

---

### Questão 5: Com ruído, qual estratégia teve o maior aumento no erro nas estações? Por quê?
**Resposta:**
1. **Identificação:** A **Lei Polar Ponto a Ponto** apresentou o maior aumento absoluto e relativo de erro de parada nas estações sob ruído, saltando de erros nominais para médias próximas do limite regulamentar ($2{,}93\text{ cm}$ e desvios de até $1{,}5^\circ$).
2. **Fundamentação:** Na lei polar, tanto o sinal de velocidade $u$ quanto a orientação $\omega$ são calculados com base direta no vetor $[\Delta x, \Delta y]$ e no ângulo $\alpha$. Sob ruído gaussiano na odometria ($\sigma = 1\text{ cm}$), quando o robô se aproxima da estação ($\rho \le 5\text{ cm}$), o ruído de $1\text{ cm}$ representa mais de 20% do sinal medido, corrompendo gravemente o cálculo do ângulo de visada. Como resultado, o controlador polar toma decisões erráticas na fase terminal de aproximação.
3. **Por que o Seguidor de Caminho é mais imune:** No seguidor de caminho da Tarefa B, a trajetória de aproximação é guiada pela reta de referência pré-definida. O erro lateral $e_y$ é ortogonal ao movimento e filtrado pelo termo $-k_y u e_y$, de modo que o ruído afeta predominantemente a leitura longitudinal, que é facilmente absorvida por uma rampa linear de desaceleração suave.

---

### Questão 6: Por que a lei polar, que funciona bem em espaço livre (Tarefa A), faz ondulações no roteiro? Em quais faixas (internas ou externas) isso é mais perigoso, e por quê? Em que situações de uma linha de montagem real cada estratégia é vantagem ou risco?
**Resposta:**
1. **Causa das ondulações da Lei Polar:** A lei polar é um controlador de *pose-to-pose*, projetado para conduzir o robô até um destino único em campo aberto. Ela não possui conceito de "faixa" ou "caminho de referência" (não faz *lane keeping*). Ao receber como meta um ponto de passagem (como a quina C1 ou uma estação subsequente), a lei polar combina rotação e translação instantânea. Por exemplo, ao sair da estação SEe voltado para $180^\circ$ rumo a NEe, a orientação atual está a $90^\circ$ da linha reta. O robô transroga e rotaciona ao mesmo tempo, descrevendo uma parábola/arco amplo para fora da linha reta antes de convergir para a orientação final. Ao encadear 17 metas, essas curvas amplas geram um movimento ondulatório oscilatório, inflando o trajeto de $57{,}2\text{ m}$ para $66{,}36\text{ m}$.
2. **Faixas mais perigosas:** As **faixas internas** (corredor central entre $x = -1{,}4\text{ m}$ e $x = 1{,}4\text{ m}$).
   - **Justificativa geométrica:** O corredor central possui apenas $2{,}8\text{ m}$ entre as faces dos blocos fabris (localizados em $x = \pm 2{,}0\text{ m}$). A estação interna fica a apenas $0{,}60\text{ m}$ de distância da parede do bloco. Quando o robô executa as ondulações características da lei polar no corredor interno, ele se aproxima perigosamente da parede dos obstáculos: a simulação acusou uma folga mínima de míseros **$0{,}009\text{ m}$** ($9\text{ mm}$), violando totalmente a exigência de segurança de $0{,}35\text{ m}$ e quase colidindo com as máquinas da fábrica!
3. **Análise de Vantagens e Riscos em Linhas de Montagem Reais:**
   - **Seguidor de Caminho (Tarefa B — Recomendado):**
     - *Vantagens:* Mantém o robô rigorosamente contido em sua faixa de tráfego (desvio máx. $2{,}06\text{ cm}$), folga aos blocos estável ($0{,}579\text{ m} > 0{,}35\text{ m}$), trânsito fluído e contínuo nas quinas sem parar, menor esforço de controle angular ($42{,}77\text{ rad}^2/\text{s}$).
     - *Riscos:* Exige pré-planejamento geométrico contínuo de caminhos tangentes.
   - **Duas Fases (Gira-Anda-Gira):**
     - *Vantagens:* Movimento geometricamente previsível (só anda em linha reta); não invade as laterais dos blocos.
     - *Riscos:* Para e gira no lugar em cada quina e estação. Isso causa elevado esforço e desgaste mecânico nos redutores e motores ($\int \omega^2 dt = 59{,}53\text{ rad}^2/\text{s}$, 39% superior ao seguidor), gera trancos na carga transportada e reduz a eficiência logística da fábrica.
   - **Lei Polar Ponto a Ponto:**
     - *Vantagens:* Excelente para manobras de estacionamento em pátios abertos sem obstáculos circundantes.
     - *Riscos:* **Inadmissível para corredores de fábrica estreitos.** O robô não respeita linhas retas e invade o espaço de segurança das máquinas e pedestres, com risco iminente de colisão.

---

## 5. Conclusão e Recomendação Técnica

Os resultados quantitativos obtidos nas quatro tarefas comprovam que o **Seguidor de Caminho Contínuo (Tarefa B)** é a solução definitiva, mais segura e eficiente para a automação da linha de montagem com o robô Pioneer 3DX:
1. Concluiu o roteiro de $57{,}1\text{ m}$ em **$196{,}7\text{ s}$** (limite: $300\text{ s}$).
2. Manteve desvio lateral de **$2{,}06\text{ cm}$** (limite: $5{,}0\text{ cm}$).
3. Garantiu folga mínima de **$0{,}579\text{ m}$** (limite: $0{,}35\text{ m}$).
4. Concluiu todas as paradas em estações com precisão de **$1{,}3\text{ cm}$** e **$0{,}7^\circ$** (limites: $3\text{ cm}$ e $2^\circ$).
5. Sobreviveu com total estabilidade a 10 testes sob ruído estocástico de odometria e atuação (Tarefa C).

A implementação encontra-se disponível no arquivo [`avaliacao_p3dx.py`](file:///c:/Users/sidgubert/Documents/dev/sistemas-autonomos-roboticos/avaliacao_p3dx.py) com todos os gráficos de trajetória e controle gerados e salvos no diretório [`figuras/`](file:///c:/Users/sidgubert/Documents/dev/sistemas-autonomos-roboticos/figuras).
