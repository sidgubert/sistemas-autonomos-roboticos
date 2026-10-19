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

O projeto contemplou quatro tarefas essenciais no mesmo programa ([`avaliacao_p3dx.py`](avaliacao_p3dx.py)):
1. **Tarefa A (Controle Contínuo até uma Pose):** Implementação e sintonia da lei de controle não-linear em coordenadas polares (Siegwart / Aicardi) com comprovação de estabilidade de Lyapunov e tratamento de alvos retrógrados ($|\alpha| > 90^\circ$).
2. **Tarefa B (Seguidor de Caminho Suave):** Síntese de um seguidor de caminho com antecipação de curvatura (*feedforward* $\kappa$) e realimentação de erro lateral ($e_y$) e erro de rumo ($e_\psi$) sobre um circuito composto por retas e arcos de raio $R = 0{,}5\text{ m}$ tangentes nas quinas C1 a C8. O robô contorna as quinas sem parar e para nas 8 estações e na Base dentro das tolerâncias estritas ($\rho < 3\text{ cm}$, $|\Delta\psi| < 2^\circ$, desvio lateral $\le 5\text{ cm}$), além de demonstrar o efeito ondulatório caso a lei polar seja aplicada diretamente ao roteiro.
3. **Tarefa C (Robustez a Ruído):** Avaliação estatística em 10 execuções independentes sob ruído de odometria gaussiano ($\sigma_{x,y} = 1\text{ cm}$, $\sigma_\psi = 0{,}5^\circ$) e ruído de atuação multiplicativo ($\mathcal{N}(0, 0{,}05)$), separando a pose verdadeira da pose estimada.
4. **Tarefa D (Análise Comparativa de Três Estratégias — Sem e Com Ruído):** Comparativo experimental quantitativo entre o **Seguidor de Caminho**, a **Lei Polar Ponto-a-Ponto** e o **Controlador em Duas Fases (Gira-Anda-Gira)** pelas 6 métricas de desempenho da Seção 5 do enunciado.

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

$$\rho = \sqrt{(x_d - x)^2 + (y_d - y)^2}, \quad \alpha = \text{atan2}(y_d - y,\, x_d - x) - \psi, \quad \beta = \psi_f - \psi - \alpha$$

com $\alpha, \beta \in [-\pi, \pi]$. A lei de controle cinemático contínua é dada por:

$$u = k_\rho \rho \max(\cos\alpha, 0), \quad \omega = k_\alpha \alpha + k_\beta \beta$$

#### Condições de Estabilidade de Lyapunov:
Pelo teorema de Aicardi et al. e Siegwart, a função candidata de Lyapunov garante estabilidade assintótica local sse:

$$k_\rho > 0, \quad k_\beta < 0, \quad k_\alpha - k_\rho > 0$$

**Ganhos adotados e validados:**
- $k_\rho = 0{,}50\text{ s}^{-1} > 0$
- $k_\alpha = 1{,}50\text{ s}^{-1}$ (satisfaz $k_\alpha - k_\rho = 1{,}00 > 0$)
- $k_\beta = -0{,}30\text{ s}^{-1} < 0$

#### Tratamento de Alvos Retrógrados ($|\alpha| > 90^\circ$):
Quando o alvo encontra-se atrás do robô ($|\alpha| > \pi/2$), a projeção $\cos\alpha < 0$. Se a lei linear direta $u = k_\rho \rho$ fosse mantida sem modulação, o robô aceleraria para a frente (afastando-se do destino) enquanto tentava virar, produzindo uma laçada frontal ampla e perigosa (*cusp*). A equipe adotou a modulação suave $u = k_\rho \rho \max(0, \cos\alpha)$, que zera suavemente a translação para alvos posteriores, priorizando a rotação até que o robô aponte para o semiespaço do alvo ($|\alpha| \le 90^\circ$).

---

### 2.3 Tarefa B — Seguidor de Caminho Contínuo (Retas e Arcos)
O caminho de referência $\Gamma(s)$ foi parametrizado rigorosamente conforme a Seção 3 do enunciado:
- Comprimento total da rota: **$57{,}08\text{ m}$** (conforme os $57{,}1\text{ m}$ previstos pelo cálculo analítico).
- Quinas C1 a C8: arcos circulares de raio $R = 0{,}5\text{ m}$ tangentes às retas de entrada e saída.
- Curvatura nominal: $\kappa = 0$ nas retas; $\kappa = +1/R = +2{,}0\text{ m}^{-1}$ em curvas anti-horárias (CCW: C1, C2, C3, C6, C7, C8); $\kappa = -1/R = -2{,}0\text{ m}^{-1}$ em curvas horárias (CW: C4, C5).

No referencial local de Serret-Frenet do segmento ativo, o erro de rumo $e_\psi$ e o erro lateral $e_y$ são dados por:

$$e_\psi = \text{atan2}(\sin(\psi_{\text{caminho}} - \psi),\, \cos(\psi_{\text{caminho}} - \psi))$$
$$e_y = -(x - x_r)\sin\psi_r + (y - y_r)\cos\psi_r$$

A lei de controle do seguidor de caminho limita $u$ antes do cálculo de $\omega$:

$$u = \text{clip}\big(u_{\text{ref}} \max(\cos e_\psi,\, 0),\, -u_{\max},\, u_{\max}\big)$$
$$\omega = u \kappa + k_\psi e_\psi - k_y u e_y$$

onde:
- $u \kappa$: componente de antecipação (*feedforward*) de curvatura, que comanda a rotação necessária para percorrer o arco de raio $R$ sem atraso de fase.
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

| Caso de Teste | Pose Alvo $(x_d, y_d, \psi_f)$ | Tempo (s) | Comp. (m) | Erro $\rho$ (cm) | Erro $\|\Delta\psi\|$ (°) | $\int u^2 dt$ ($\text{m}^2/\text{s}$) | $\int \omega^2 dt$ ($\text{rad}^2/\text{s}$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1) Destino à frente** | $(3{,}0\text{ m},\, 1{,}0\text{ m},\, 0^\circ)$ | $10{,}6$ | $3{,}16$ | $2{,}89\text{ cm}$ | $1{,}44^\circ$ | $1{,}84$ | $0{,}15$ | **CONVERGIU** |
| **2) Destino ao lado** | $(0{,}0\text{ m},\, 2{,}0\text{ m},\, 90^\circ)$ | $9{,}6$ | $2{,}16$ | $2{,}87\text{ cm}$ | $1{,}62^\circ$ | $1{,}02$ | $2{,}20$ | **CONVERGIU** |
| **3) Destino atrás** | $(-2{,}0\text{ m},\, 0{,}0\text{ m},\, 180^\circ)$ | $10{,}5$ | $2{,}16$ | $2{,}87\text{ cm}$ | $1{,}62^\circ$ | $1{,}02$ | $4{,}94$ | **CONVERGIU** |

*Critério de aceitação:* $\rho < 3{,}0\text{ cm}$ e $|\Delta\psi| < 2{,}0^\circ$ — **100% cumprido**.

---

### 3.2 Resultados da Tarefa B (Roteiro Completo Sem Ruído)

- **1. Tempo Total:** $196{,}7\text{ s}$ (Requisito: $\le 300\text{ s}$) $\to$ Margem temporal de $34{,}4\%$.
- **2. Comprimento Percorrido:** $57{,}21\text{ m}$ (Nominal: $57{,}1\text{ m}$) $\to$ Rastreamento estrito sem desvios parasitas.
- **3. Erro Médio de Pose nas Estações:** $\rho = 1{,}31\text{ cm}$ ($< 3{,}0\text{ cm}$) e $|\Delta\psi| = 0{,}73^\circ$ ($< 2{,}0^\circ$).
- **4. Desvio da Referência:** Máximo = **$2{,}06\text{ cm}$** (Requisito: $\le 5{,}0\text{ cm}$); RMS = **$0{,}68\text{ cm}$**.
- **5. Folga Mínima aos Obstáculos:** **$0{,}579\text{ m}$** (Requisito: $\ge 0{,}35\text{ m}$) $\to$ Aprovado com margem de $22{,}9\text{ cm}$ acima do limite de segurança.
- **6. Esforço de Controle:** $\int u^2 dt = 22{,}73\text{ m}^2/\text{s}$, $\int \omega^2 dt = 42{,}77\text{ rad}^2/\text{s}$.

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

Sob perturbações simultâneas de odometria estocástica ($\sigma_{x,y} = 1\text{ cm}$, $\sigma_\psi = 0{,}5^\circ$) e imperfeição multiplicativa nos atuadores ($\mathcal{N}(0, 0{,}05)$), calculando as métricas sobre a **pose verdadeira** do simulador:

| Métrica Oficial (Seção 5) | Valor Médio $\pm$ Desvio Padrão | Unidade | Valor Exigido | Situação |
| :--- | :---: | :---: | :---: | :---: |
| **1. Tempo total** | **$194{,}500 \pm 0{,}645$** | $\text{s}$ | $\le 300\text{ s}$ | **Atende** |
| **2. Comprimento percorrido** | **$57{,}207 \pm 0{,}019$** | $\text{m}$ | Ref. $\approx 57{,}1\text{ m}$ | **Atende** |
| **3. Erro de pose (distância $\rho$)** | **$1{,}648 \pm 0{,}258$** | $\text{cm}$ | $< 3{,}0\text{ cm}$ | **Atende** |
| **3. Erro de pose (orientação $\|\Delta\psi\|$)** | **$0{,}867 \pm 0{,}078$** | $^\circ$ | $< 2{,}0^\circ$ | **Atende** |
| **4. Desvio da referência (Máximo)** | **$2{,}474 \pm 0{,}227$** | $\text{cm}$ | $\le 5{,}0\text{ cm}$ | **Atende** |
| **4. Desvio da referência (RMS)** | **$0{,}748 \pm 0{,}046$** | $\text{cm}$ | — | Excelente |
| **5. Folga mínima a obstáculos** | **$0{,}579 \pm 0{,}003$** | $\text{m}$ | $\ge 0{,}35\text{ m}$ | **Atende** |
| **6. Esforço linear ($\int u^2 dt$)** | **$22{,}785 \pm 0{,}025$** | $\text{m}^2/\text{s}$ | — | Otimizado |
| **6. Esforço angular ($\int \omega^2 dt$)** | **$43{,}638 \pm 0{,}162$** | $\text{rad}^2/\text{s}$ | — | Otimizado |

---

### 3.4 Resultados da Tarefa D (Comparação de Estratégias — Sem e Com Ruído)

#### Tabela D.1 — Comparativo Sem Ruído:

| Métrica de Desempenho (Seção 5) | Seguidor de Caminho (Tarefa B) | Lei Polar Ponto a Ponto | Duas Fases (Gira-Anda-Gira) |
| :--- | :---: | :---: | :---: |
| **1. Tempo Total (s)** | $196{,}7\text{ s}$ | $209{,}7\text{ s}$ | $\mathbf{184{,}1\text{ s}}$ |
| **2. Comprimento Percorrido (m)** | $\mathbf{57{,}21\text{ m}}$ | $66{,}36\text{ m}$ (+16%) | $57{,}69\text{ m}$ |
| **3. Erro médio $\rho$ estações (cm)** | $\mathbf{1{,}31\text{ cm}}$ | $2{,}93\text{ cm}$ | $2{,}23\text{ cm}$ |
| **3. Erro médio $\|\Delta\psi\|$ estações (°)** | $\mathbf{0{,}73^\circ}$ | $1{,}13^\circ$ | $1{,}29^\circ$ |
| **4. Desvio da referência Máx. (cm)** | $\mathbf{2{,}06\text{ cm}}$ | $125{,}65\text{ cm}$ (ondulações) | $12{,}68\text{ cm}$ (quinas vivas) |
| **4. Desvio da referência RMS (cm)** | $\mathbf{0{,}68\text{ cm}}$ | $36{,}29\text{ cm}$ | $4{,}82\text{ cm}$ |
| **5. Folga Mínima a Obstáculos (m)** | $\mathbf{0{,}579\text{ m}}$ | **$0{,}009\text{ m}$ (VIOLA!)** | $0{,}519\text{ m}$ |
| **6. Esforço Linear $\int u^2 dt$ ($\text{m}^2/\text{s}$)** | $\mathbf{22{,}73}$ | $37{,}02$ | $25{,}39$ |
| **6. Esforço Angular $\int \omega^2 dt$ ($\text{rad}^2/\text{s}$)** | **$42{,}77$** | $25{,}11$ | $59{,}53$ (+39%) |

#### Tabela D.2 — Comparativo Com Ruído (Odometria + Atuação, Semente = 42):

| Métrica de Desempenho (Seção 5) | Seguidor de Caminho (Tarefa B) | Lei Polar Ponto a Ponto | Duas Fases (Gira-Anda-Gira) |
| :--- | :---: | :---: | :---: |
| **1. Tempo Total (s)** | $195{,}3\text{ s}$ | $204{,}7\text{ s}$ | $\mathbf{184{,}0\text{ s}}$ |
| **2. Comprimento Percorrido (m)** | $\mathbf{57{,}20\text{ m}}$ | $66{,}32\text{ m}$ | $57{,}85\text{ m}$ |
| **3. Erro médio $\rho$ estações (cm)** | $\mathbf{1{,}60\text{ cm}}$ (+0,29 cm) | **$3{,}72\text{ cm}$ (+0,79 cm — VIOLA!)** | $2{,}35\text{ cm}$ (+0,12 cm) |
| **3. Erro médio $\|\Delta\psi\|$ estações (°)** | $\mathbf{0{,}61^\circ}$ | $1{,}13^\circ$ | $1{,}21^\circ$ |
| **4. Desvio da referência Máx. (cm)** | $\mathbf{3{,}06\text{ cm}}$ | $124{,}16\text{ cm}$ | $14{,}65\text{ cm}$ |
| **4. Desvio da referência RMS (cm)** | $\mathbf{0{,}77\text{ cm}}$ | $36{,}80\text{ cm}$ | $4{,}82\text{ cm}$ |
| **5. Folga Mínima a Obstáculos (m)** | $\mathbf{0{,}579\text{ m}}$ | **$0{,}012\text{ m}$ (VIOLA!)** | $0{,}536\text{ m}$ |
| **6. Esforço Linear $\int u^2 dt$ ($\text{m}^2/\text{s}$)** | $\mathbf{22{,}75}$ | $36{,}76$ | $25{,}60$ |
| **6. Esforço Angular $\int \omega^2 dt$ ($\text{rad}^2/\text{s}$)** | **$43{,}84$** | $27{,}32$ | $64{,}82$ (+48%) |

---

## 4. Respostas Aprofundadas às Questões Orientadoras

### Questão 1: Como os ganhos $k_\psi$ e $k_y$ do seguidor de caminho afetam o desvio em relação às retas? O que acontece sem a correção lateral ($k_y = 0$) e com $k_y$ muito alto?
**Resposta:**
O termo angular $\omega = u\kappa + k_\psi e_\psi - k_y u e_y$ atua em duas variáveis de estado acopladas da cinemática de Serret-Frenet: a orientação do robô ($\dot{e}_\psi \approx -\omega + u\kappa$) e o deslocamento lateral ($\dot{e}_y = u \sin e_\psi \approx u e_\psi$).
1. **Sem correção lateral ($k_y = 0$):** O laço fecha apenas sobre a orientação ($e_\psi \to 0$). Se o robô possuir qualquer erro de posição inicial ao sair de uma curva ou sofrer distúrbio transitório, ele alinhará sua orientação paralelamente à reta de referência, mas navegará com um deslocamento perpendicular constante e não-nulo ($e_y \neq 0$). Sem realimentação de $e_y$, o sistema possui um integrador aberto na posição transversal, sendo incapaz de anular erros de rastreamento lateral.
2. **Com $k_y$ excessivamente alto:** O polo do sistema em malha fechada é empurrado para frequências elevadas. Para pequenos erros laterais, o ganho comanda variações abruptas de $\omega$, levando à saturação frequente de velocidade angular ($|\omega| = 100^\circ/\text{s}$). Como a frequência de controle é de $10\text{ Hz}$ ($\Delta t = 0{,}1\text{ s}$), essa hipercorreção produz atraso de fase, originando sobre-elevação (*overshoot*) e oscilações em zigue-zague ao redor da reta (*hunting effect*), aumentando o erro RMS e o esforço de controle angular.
3. **Sintonia ótima:** A combinação $k_\psi = 5{,}0$ e $k_y = 10{,}0$ estabeleceu um amortecimento bem comportado ($\zeta \approx 0{,}7$ a $0{,}8$), assegurando convergência monotônica rápida sem ondulações e mantendo o desvio máximo em $2{,}06\text{ cm}$ ($\le 5{,}0\text{ cm}$).

---

### Questão 2: A saturação de $u$ e $\omega$ altera a garantia de estabilidade? Por que, no seguidor de caminho, convém limitar $u$ antes de calcular $\omega$?
**Resposta:**
1. **Efeito da saturação na estabilidade:** Sim. As provas formais de estabilidade de Lyapunov assumem que os atuadores possuem autoridade linear irrestrita. Quando ocorre saturação desregulada (componente a componente via *clipping* independente), a razão cinemática $u/\omega$ é corrompida. Se $\omega$ saturar e $u$ não, o raio de giro instantâneo $R = u/\omega$ cresce indevidamente, fazendo o robô abrir a curva e afastar-se do caminho de referência, o que pode provocar perda de estabilidade ou colisão. A técnica adotada pela equipe de **saturar $u$ e $\omega$ proporcionalmente pelo mesmo fator de escala $s$** preserva rigorosamente a geometria da trajetória ($R = u/\omega$), transformando a saturação apenas em uma reescala temporal do avanço e mantendo a convergência geométrica.
2. **Por que limitar $u$ antes de calcular $\omega$:** Na lei de controle do seguidor de caminho ($\omega = u\kappa + k_\psi e_\psi - k_y u e_y$), dois termos dependem diretamente de $u$: o termo *feedforward* de curvatura ($u \kappa$) e o termo de realimentação lateral ($-k_y u e_y$). Se $\omega$ fosse calculado utilizando uma velocidade de referência teórica elevada, mas depois o atuador linear saturasse para um $u$ efetivo menor, haveria um descasamento cinemático (*oversteering*): o robô giraria na velocidade angular correspondente a uma velocidade linear que ele não está executando. Limitar $u$ antes de calcular $\omega$ garante consistência cinemática instantânea entre avanço longitudinal e correção angular.

---

### Questão 3: Por que a lei polar não é definida em $\rho = 0$? Como a equipe evitou oscilações perto das estações?
**Resposta:**
1. **Singularidade em $\rho = 0$:** As coordenadas polares são dadas por $\alpha = \text{atan2}(\Delta y, \Delta x) - \psi$ e $\beta = \psi_f - \psi - \alpha$. Quando o robô atinge exatamente o alvo ($\Delta x = 0, \Delta y = 0$), tem-se $\rho = 0$. Nesse ponto, a função $\text{atan2}(0, 0)$ é matematicamente indeterminada, pois não existe vetor diretor da linha de visada.
2. **Origem das oscilações terminais:** Conforme $\rho \to 0$, variações milimétricas de posição causadas por discretização temporal ($10\text{ Hz}$) ou ruído de odometria ($\sigma = 1\text{ cm}$) fazem o vetor $[\Delta x, \Delta y]$ girar abruptamente entre quadrantes opostos, variando $\text{atan2}$ em até $\pm 180^\circ$ e provocando chaveamentos bruscos de $\omega$.
3. **Solução implementada:** A equipe implementou uma máquina de estados com chaveamento na vizinhança de chegada: ao entrar no raio de tolerância da estação, a translação é desativada ($u = 0$), desacoplando a posição da orientação, e aciona-se o alinhamento angular puro $\omega = k_w \text{wrap}(\psi_f - \psi)$ até $|\psi_f - \psi| < 1^\circ$, parando completamente o robô ($\rho \approx 1{,}3\text{ cm} < 3\text{ cm}$ e $|\Delta\psi| \approx 0{,}73^\circ < 2^\circ$).

---

### Questão 4: Qual é o efeito do raio das curvas nas quinas sobre o tempo total, o desvio do caminho e a velocidade angular? Que limite físico do P3DX restringe o raio mínimo?
**Resposta:**
1. **Efeito no tempo total e velocidade angular:** Em um arco de raio $R$, a velocidade angular de equilíbrio é $\omega = u / R$. Um raio muito pequeno exige $\omega$ elevado ou redução severa de $u$ para não saturar $\omega_{\max}$. Com $R = 0{,}5\text{ m}$ e $u_{\text{arc}} = 0{,}32\text{ m/s}$, a velocidade angular nominal na curva é $\omega = 0{,}64\text{ rad/s} \approx 36{,}7^\circ/\text{s}$ (bem abaixo do limite de $100^\circ/\text{s}$), permitindo contornar as 8 quinas sem parar e encurtando a rota de $58{,}8\text{ m}$ (polilinha) para $57{,}1\text{ m}$.
2. **Efeito no desvio do caminho:** A inserção de arcos tangentes garante continuidade geométrica $G^1$ (direção da tangente contínua) nas transições reta-arco-reta. Enquanto na polilinha com quinas vivas de $90^\circ$ o controlador em duas fases apresenta desvio máximo de $12{,}68\text{ cm}$ em relação ao caminho suave, o seguidor de caminho com arcos tangentes de $R = 0{,}5\text{ m}$ mantém o desvio máximo em apenas $2{,}06\text{ cm}$.
3. **Limite físico restritivo:** O limite físico no modelo cinemático do Pioneer 3DX é dado pela velocidade angular máxima $|\omega| \le \omega_{\max} = 100^\circ/\text{s} \approx 1{,}7453\text{ rad/s}$ combinada com a velocidade linear desejada $u > 0$ sem parar nas quinas:
   $$R_{\min} = \frac{u}{\omega_{\max}}$$
   Para $u = 0{,}32\text{ m/s}$, o raio mínimo absoluto sem saturar o atuador angular (mesmo sem margem para correção de erros $e_\psi$ e $e_y$) seria $R_{\min} \approx 0{,}183\text{ m}$. Considerando a reserva de autoridade de controle para corrigir perturbações e a geometria do chassi/bitola ($L \approx 0{,}33\text{ m}$), $R = 0{,}5\text{ m}$ oferece ampla margem dinâmica.

---

### Questão 5: Com ruído, qual estratégia teve o maior aumento no erro nas estações? Por quê?
**Resposta:**
1. **Identificação com dados:** Conforme as Tabelas D.1 e D.2, a **Lei Polar Ponto a Ponto** apresentou o maior aumento no erro de posição nas estações sob ruído: seu erro médio $\rho$ subiu de **$2{,}93\text{ cm}$** (sem ruído) para **$3{,}72\text{ cm}$** (com ruído), um aumento de **$+0{,}79\text{ cm}$ ($+27{,}0\%$)**, violando a tolerância máxima de $3{,}0\text{ cm}$. Em comparação, o Seguidor de Caminho subiu apenas $+0{,}29\text{ cm}$ (de $1{,}31\text{ cm}$ para $1{,}60\text{ cm}$) e o Duas Fases subiu $+0{,}12\text{ cm}$ (de $2{,}23\text{ cm}$ para $2{,}35\text{ cm}$), ambos permanecendo aprovados com folga.
2. **Explicação física e matemática:** Na lei polar, os ângulos $\alpha = \text{atan2}(\Delta y, \Delta x) - \psi$ e $\beta = \psi_f - \psi - \alpha$ são recalculados a cada ciclo diretamente a partir das diferenças $\Delta x$ e $\Delta y$ contaminadas pelo ruído de odometria ($\sigma_{x,y} = 1\text{ cm}$). Quando o robô está próximo da estação ($\rho \approx 3\text{ a } 5\text{ cm}$), a relação sinal-ruído deteriora-se drasticamente e o termo $k_\beta \beta$ (que tenta impor a orientação final $\psi_f$ perpendicular ao corredor enquanto o robô ainda translada) induz desvios laterais e paradas prematuras induzidas por picos de ruído na estimativa de $\rho$. Já no Seguidor de Caminho, a aproximação ocorre alinhada sobre a reta de referência com desaceleração longitudinal suave antes do giro final no ponto da estação.

---

### Questão 6: Por que a lei polar, que funciona bem em espaço livre (Tarefa A), faz ondulações no roteiro? Em quais faixas (internas ou externas) isso é mais perigoso, e por quê? Em que situações de uma linha de montagem real cada estratégia é vantagem ou risco?
**Resposta:**
1. **Causa das ondulações da Lei Polar:** A lei polar controla apenas a convergência assintótica para uma pose final isolada $(x_d, y_d, \psi_f)$, sem qualquer restrição de caminho intermediário. Nas estações da linha de montagem, a orientação de atendimento $\psi_f$ é perpendicular ($90^\circ$) à direção do corredor (voltada para a face do bloco). Assim, ao partir de uma estação (ex.: `SEe` voltado para $180^\circ$) rumo à próxima (`NEe`, ao norte, também com $\psi_f = 180^\circ$), o robô avança inicialmente na direção em que está apontando e descreve um grande arco lateral (desvio máximo de **$125{,}65\text{ cm}$** da reta de referência e comprimento total inflado para **$66{,}36\text{ m}$**) para conseguir chegar ao destino apontando perpendicularmente ao corredor.
2. **Faixas mais perigosas:** As **faixas internas (corredor central, $x = \pm 1{,}4\text{ m}$)** são muito mais perigosas.
   - **Por quê:** Nas faixas externas ($x = \pm 5{,}0\text{ m}$), a linha de tráfego dista **$1{,}0\text{ m}$** da face externa dos blocos ($x = \pm 4{,}0\text{ m}$). Já no corredor central ($x = \pm 1{,}4\text{ m}$), a linha de tráfego dista apenas **$0{,}60\text{ m}$** da face interna dos blocos ($x = \pm 2{,}0\text{ m}$). Quando o robô sai de uma estação interna (onde estava parado apontando diretamente para o bloco a $0{,}60\text{ m}$ de distância) sob a lei polar, o arco inicial de manobra avança na direção do bloco, reduzindo a folga mínima para apenas **$0{,}009\text{ m}$ ($9\text{ mm}$)** e violando criticamente a margem de segurança de $0{,}35\text{ m}$.
3. **Vantagens e Riscos em uma Linha de Montagem Real:**
   - **Seguidor de Caminho (Tarefa B — Escolha da Equipe):**
     - *Vantagens:* Mantém o robô estritamente dentro do corredor de tráfego (desvio máx. $2{,}06\text{ cm}$ sem ruído e $2{,}47\text{ cm}$ médio com ruído), garante a maior folga aos blocos ($0{,}579\text{ m} \ge 0{,}35\text{ m}$), faz curvas suaves nas quinas sem parar e apresenta esforço angular $28\%$ menor que o controlador em duas fases ($42{,}77$ vs $59{,}53\text{ rad}^2/\text{s}$).
     - *Riscos/Requisitos:* Requer definição prévia da geometria do caminho (retas e arcos tangentes).
   - **Controlador em Duas Fases (Gira-Anda-Gira):**
     - *Vantagens:* Simples de implementar e mantém o robô próximo às retas nos corredores (folga de $0{,}519\text{ m}$).
     - *Riscos:* Obriga o robô a parar e girar no próprio eixo em todas as quinas C1–C8, gerando descontinuidade de velocidade, desgaste mecânico acentuado nos pneus e redutores ($\int \omega^2 dt = 59{,}53\text{ rad}^2/\text{s}$ sem ruído e $64{,}82\text{ rad}^2/\text{s}$ com ruído), risco de instabilidade da carga transportada e desvio geométrico de $12{,}68\text{ cm}$ nas quinas.
   - **Lei Polar Ponto a Ponto:**
     - *Vantagens:* Útil para manobras em pátios abertos e desobstruídos (como na Tarefa A).
     - *Riscos:* **Inviável e perigosa em corredores industriais**, pois gera ondulações de $1{,}25\text{ m}$ fora da faixa, invade a zona de segurança dos blocos ($0{,}009\text{ m} < 0{,}35\text{ m}$) e perde precisão de parada sob ruído ($3{,}72\text{ cm} > 3{,}0\text{ cm}$).

---

## 5. Conclusão e Recomendação Técnica

Com base nos dados experimentais das quatro tarefas, a equipe recomenda o **Seguidor de Caminho Contínuo (Tarefa B)** para operação na linha de montagem real:
1. Concluiu os 17 trechos ($57{,}21\text{ m}$) em **$196{,}7\text{ s}$** ($\le 300\text{ s}$).
2. Manteve desvio máximo da rota em **$2{,}06\text{ cm}$** sem ruído e **$2{,}47 \pm 0{,}23\text{ cm}$** com ruído ($\le 5{,}0\text{ cm}$).
3. Garantiu folga mínima aos obstáculos de **$0{,}579\text{ m}$** em todas as execuções ($\ge 0{,}35\text{ m}$).
4. Parou nas 8 estações e na Base com erro médio de **$1{,}31\text{ cm}$** e **$0{,}73^\circ$** sem ruído, e **$1{,}65 \pm 0{,}26\text{ cm}$** e **$0{,}87 \pm 0{,}08^\circ$** nas 10 execuções com ruído ($< 3\text{ cm}$ e $< 2^\circ$).
