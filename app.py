"""
Como se move o triângulo?
Tarefa exploratória sobre transformações no plano com matrizes (Streamlit).

Módulos autónomos:  A · variações de tamanho (scaling)  ·  B · cisalhamentos (shearing)  ·  C · rotações  ·  D · exercício (em construção).

Executar:  streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import streamlit as st

st.set_page_config(page_title="Como se move o triângulo?", page_icon="📐", layout="wide")

# --------------------------------------------------------------------------
# Dados e funções auxiliares
# --------------------------------------------------------------------------
# Matriz X dos vértices do polígono: a coluna i é o vértice x_i = (x_i; y_i) -> A(1,1), B(4,1), C(2,3)
X = np.array([[1, 4, 2],
              [1, 1, 3]], dtype=float)

CORES = ["tab:blue", "tab:red", "tab:green", "tab:orange"]


def _eixos(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)


def desenhar(figuras, nomes, alvo=None, lim=9, rotulos=True, circulos=False):
    """Devolve uma figura matplotlib com os polígonos (matrizes 2×n, colunas = vértices).

    rotulos:  A, B, C (original), A′, B′, C′ (1.ª imagem), A″, B″, C″ (2.ª imagem)...
    circulos: circunferências com centro na origem que passam pelos vértices da 1.ª figura.
    """
    fig = Figure(figsize=(5.5, 5.5))  # Figure direto (sem pyplot): seguro com vários utilizadores em simultâneo
    ax = fig.subplots()
    if circulos:
        t = np.linspace(0, 2 * np.pi, 200)
        for j in range(figuras[0].shape[1]):
            r = np.hypot(figuras[0][0][j], figuras[0][1][j])
            ax.plot(r * np.cos(t), r * np.sin(t), color="0.7", lw=1, ls=":", zorder=0)
    for i, (P, nome) in enumerate(zip(figuras, nomes)):
        xs, ys = np.append(P[0], P[0][0]), np.append(P[1], P[1][0])
        ax.fill(xs, ys, alpha=0.2, color=CORES[i % 4])
        ax.plot(xs, ys, "o-", color=CORES[i % 4], label=nome)
        if rotulos:
            sufixo = "′" * i if i < 3 else f"({i})"
            for j in range(P.shape[1]):
                ax.annotate(f"{chr(65 + j)}{sufixo}", (P[0][j], P[1][j]), textcoords="offset points",
                            xytext=(7, 7), color=CORES[i % 4], fontsize=11, fontweight="bold")
    if alvo is not None:
        xs, ys = np.append(alvo[0], alvo[0][0]), np.append(alvo[1], alvo[1][0])
        ax.plot(xs, ys, "k--", label="alvo")
    _eixos(ax, lim)
    ax.legend(loc="upper left")
    return fig


def desenhar_base(M):
    """Quadrado unitário e a sua imagem, com as setas de e1, e2 e das suas imagens (colunas de M)."""
    Q = np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float)
    Q1 = M @ Q
    lim = max(2.0, float(np.ceil(np.abs(np.hstack([Q, Q1])).max())) + 1)
    fig = Figure(figsize=(5.5, 5.5))
    ax = fig.subplots()
    for P, cor, nome in ((Q, "tab:blue", "quadrado unitário"), (Q1, "tab:red", "imagem")):
        xs, ys = np.append(P[0], P[0][0]), np.append(P[1], P[1][0])
        ax.fill(xs, ys, alpha=0.15, color=cor)
        ax.plot(xs, ys, "-", color=cor, label=nome)
    for k in range(2):
        for w, cor, rot in ((np.eye(2)[:, k], "tab:blue", rf"$\mathbf{{e}}_{k + 1}$"),
                            (M[:, k], "tab:red", rf"$M\mathbf{{e}}_{k + 1}$")):
            ax.annotate("", xy=w, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=cor, lw=2))
            ax.annotate(rot, w, textcoords="offset points", xytext=(6, 6), color=cor, fontsize=12)
    _eixos(ax, lim)
    ax.legend(loc="upper left")
    return fig


def mostrar(fig):
    """Mostra a figura como PNG a 120 dpi (mais leve e rápido do que o st.pyplot, que grava a 200 dpi)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    st.image(buf.getvalue())


def area(P):
    """Área do polígono (fórmula dos cordões)."""
    x, y = P[0], P[1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def lados(P):
    """Comprimentos dos lados, pela ordem dos vértices (AB, BC, CA)."""
    return np.linalg.norm(P - np.roll(P, -1, axis=1), axis=0)


def angulos_internos(P):
    """Ângulos internos (em graus) em cada vértice; None se a figura for degenerada."""
    n = P.shape[1]
    res = []
    for i in range(n):
        a = P[:, (i - 1) % n] - P[:, i]
        b = P[:, (i + 1) % n] - P[:, i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return None
        res.append(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))
    return np.array(res)


def mostrar_medidas(P, P1, rotacao=False):
    """Compara a figura P com a sua imagem P1 (área, lados, ângulos internos)."""
    a0, a1 = area(P), area(P1)
    cols = st.columns(4 if rotacao else 3)
    cols[0].metric("Área (original → imagem)", f"{a0:.2f} → {a1:.2f}",
                   f"razão {a1 / a0:.3f}" if a0 else None, delta_color="off")
    cols[1].write("**Lados** (AB, BC, CA)")
    cols[1].write(f"{np.round(lados(P), 3).tolist()}  \n→ {np.round(lados(P1), 3).tolist()}")
    g0, g1 = angulos_internos(P), angulos_internos(P1)
    cols[2].write("**Ângulos internos** (A, B, C)")
    cols[2].write(f"{np.round(g0, 1).tolist()}°  \n→ "
                  + (f"{np.round(g1, 1).tolist()}°" if g1 is not None else "— (figura degenerada)"))
    if rotacao:
        ang = np.degrees(np.arctan2(P1[1, 0], P1[0, 0]) - np.arctan2(P[1, 0], P[0, 0])) % 360
        cols[3].metric("Ângulo O→A até O→A′", f"{ang:.1f}°",
                       help="Medido em graus, no sentido contrário ao dos ponteiros do relógio.")


def _f(v):
    """Número formatado para LaTeX (4 casas, sem «-0»)."""
    return f"{round(float(v), 4) + 0.0:g}"


def latex_mat(A):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    linhas = r" \\ ".join(" & ".join(_f(v) for v in linha) for linha in A)
    return r"\begin{bmatrix} " + linhas + r" \end{bmatrix}"


latex_matriz = latex_mat


def latex_coluna(v):
    return latex_mat(np.asarray(v, dtype=float).reshape(-1, 1))


def latex_produto(M, P):
    return rf"{latex_mat(M)}{latex_mat(P)} = {latex_mat(M @ P)}"


def tabela_md(df):
    """Converte um DataFrame numa tabela Markdown (sem dependências extra)."""
    cols = list(df.columns)
    linhas = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        linhas.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in cols) + " |")
    return "\n".join(linhas)


def entrada_matriz(prefixo, M0=((1.0, 0.0), (0.0, 1.0))):
    """Caixas numéricas para a matriz M (2×2)."""
    st.markdown("**Matriz M**")
    c1, c2 = st.columns(2)
    p = c1.number_input("p", value=float(M0[0][0]), step=0.1, key=f"{prefixo}_p")
    q = c2.number_input("q", value=float(M0[0][1]), step=0.1, key=f"{prefixo}_q")
    r = c1.number_input("r", value=float(M0[1][0]), step=0.1, key=f"{prefixo}_r")
    s = c2.number_input("s", value=float(M0[1][1]), step=0.1, key=f"{prefixo}_s")
    return np.array([[p, q], [r, s]])


def sliders_matriz(prefixo):
    """Seletores para a matriz M (2×2), começando na identidade."""
    p = st.slider("p", -3.0, 3.0, 1.0, 0.1, key=f"{prefixo}_p")
    q = st.slider("q", -3.0, 3.0, 0.0, 0.1, key=f"{prefixo}_q")
    r = st.slider("r", -3.0, 3.0, 0.0, 0.1, key=f"{prefixo}_r")
    s = st.slider("s", -3.0, 3.0, 1.0, 0.1, key=f"{prefixo}_s")
    return np.array([[p, q], [r, s]])


def matriz_rotacao(alfa):
    return np.array([[np.cos(alfa), -np.sin(alfa)],
                     [np.sin(alfa), np.cos(alfa)]])


def entrada_rotacao(prefixo, c0=0.0, slider=False):
    """O input é o valor de c, com o ângulo de rotação α = c·π radianos. Devolve M(α)."""
    rotulo = "c  (ângulo de rotação α = c·π radianos)"
    if slider:
        c = st.slider(rotulo, -2.0, 2.0, float(c0), 0.05, key=f"{prefixo}_c")
    else:
        c = st.number_input(rotulo, value=float(c0), step=0.05, format="%.3f", key=f"{prefixo}_c")
    alfa = c * np.pi
    st.caption(f"α = {c:g}·π ≈ {alfa:.4f} rad")
    M = matriz_rotacao(alfa)
    st.markdown("Cada elemento de $M$ é uma **função do ângulo** $\\alpha$:")
    st.latex(r"M(\alpha) = \begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix}")
    st.markdown("Para o valor de $\\alpha$ escolhido:")
    st.latex(rf"\cos\alpha = {_f(np.cos(alfa))}, \qquad \sin\alpha = {_f(np.sin(alfa))}")
    st.latex(rf"M = {latex_mat(M)}")
    return M


def mostrar_composicao(M1, M2, circulos=False):
    """Aplica M1 e depois M2 à figura e mostra o produto M2·M1. Devolve M2·M1."""
    X1 = M1 @ X
    X2 = M2 @ X1
    mostrar(desenhar([X, X1, X2], ["original", "após a 1.ª", "após a 2.ª"], circulos=circulos))
    P = M2 @ M1
    st.latex(rf"M_2\,M_1 = {latex_mat(M2)}{latex_mat(M1)} = {latex_mat(P)}")
    st.caption("Primeiro aplica-se $M_1$ e depois $M_2$: $\\mathbf{x}'' = M_2\\,(M_1\\,\\mathbf{x}) = (M_2\\,M_1)\\,\\mathbf{x}$.")
    return P


def justificar_comum(M):
    """Colunas de M, produto M·X, determinante e quadrado unitário."""
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("**As colunas de $M$ são as imagens de $\\mathbf{e}_1 = (1,0)$ e $\\mathbf{e}_2 = (0,1)$:**")
        st.latex(rf"M\mathbf{{e}}_1 = {latex_coluna(M @ e1)}, \qquad M\mathbf{{e}}_2 = {latex_coluna(M @ e2)}")
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        st.markdown("**Determinante e área:**")
        st.latex(rf"\det M = ps - qr = {_f(det)}")
        st.write(f"Razão das áreas (imagem / original): **{area(M @ X) / area(X):.3f}**")
    with c2:
        mostrar(desenhar_base(M))
    st.markdown("**O produto, coluna a coluna:** $X' = M\\,X$")
    st.latex(rf"M\,X = {latex_produto(M, X)}")


# --------------------------------------------------------------------------
# Desafios de cada módulo (a solução só é mostrada no modo professor)
# --------------------------------------------------------------------------
DESAFIOS_SC = [
    {
        "titulo": "1. Ampliar para o dobro",
        "texto": "Amplia a figura para o dobro, **mantendo a forma**.",
        "alvo": 2 * X,
        "M_sol": np.array([[2.0, 0.0], [0.0, 2.0]]),
        "solucao": r"$M = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$.",
    },
    {
        "titulo": "2. Reduzir para metade",
        "texto": "Reduz a figura para metade, **mantendo a forma**.",
        "alvo": 0.5 * X,
        "M_sol": np.array([[0.5, 0.0], [0.0, 0.5]]),
        "solucao": r"$M = \begin{bmatrix} 0{,}5 & 0 \\ 0 & 0{,}5 \end{bmatrix}$.",
    },
    {
        "titulo": "3. Alargar na horizontal",
        "texto": "Alarga a figura na horizontal (fator 2) e deixa-a igual na vertical.",
        "alvo": np.array([[2, 8, 4], [1, 1, 3]], dtype=float),
        "M_sol": np.array([[2.0, 0.0], [0.0, 1.0]]),
        "solucao": r"$M = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$.",
    },
    {
        "titulo": "4. Alongar na vertical",
        "texto": "Alonga a figura na vertical (fator 1,5) e deixa-a igual na horizontal.",
        "alvo": np.array([[1, 4, 2], [1.5, 1.5, 4.5]], dtype=float),
        "M_sol": np.array([[1.0, 0.0], [0.0, 1.5]]),
        "solucao": r"$M = \begin{bmatrix} 1 & 0 \\ 0 & 1{,}5 \end{bmatrix}$.",
    },
    {
        "titulo": "5. Refletir em Oy",
        "texto": "Reflete a figura em relação ao eixo $Oy$. *(Isto ainda é scaling? Discute.)*",
        "alvo": np.array([[-1, -4, -2], [1, 1, 3]], dtype=float),
        "M_sol": np.array([[-1.0, 0.0], [0.0, 1.0]]),
        "solucao": r"$M = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$ (scaling com fator negativo).",
    },
    {
        "titulo": "6. Meia volta, só com scaling",
        "texto": "Consegues obter, **só com uma matriz de scaling**, a figura que resulta de uma rotação de meia volta em torno da origem?",
        "alvo": -X,
        "M_sol": np.array([[-1.0, 0.0], [0.0, -1.0]]),
        "solucao": r"$M = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}$ (os dois fatores são negativos).",
    },
    {
        "titulo": "7. Desafio aberto",
        "texto": (
            "Encontra uma matriz de scaling que **duplique a área** da figura, **sem manter a forma** "
            "(ou seja, com fatores diferentes). Usa a medição da razão das áreas para confirmares."
        ),
        "alvo": None,
        "solucao": r"Qualquer $\begin{bmatrix} k & 0 \\ 0 & l \end{bmatrix}$ com $|k\,l| = 2$ e $|k| \neq |l|$, por exemplo $k = 2$, $l = 1$ ou $k = 4$, $l = 0{,}5$.",
    },
]

DESAFIOS_SH = [
    {
        "titulo": "1. Inclinar para a direita",
        "texto": "Inclina a figura \"para a direita\", de modo que cada vértice se desloque na horizontal tanto quanto a sua altura ($y$).",
        "alvo": np.array([[2, 5, 5], [1, 1, 3]], dtype=float),
        "M_sol": np.array([[1.0, 1.0], [0.0, 1.0]]),
        "solucao": r"$M = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$, ou seja, $x' = x + y$ e $y' = y$.",
    },
    {
        "titulo": "2. Inclinar para a esquerda",
        "texto": "Inclina a figura \"para a esquerda\", de modo que cada vértice se desloque na horizontal metade da sua altura ($y$), para a esquerda.",
        "alvo": np.array([[0.5, 3.5, 0.5], [1, 1, 3]], dtype=float),
        "M_sol": np.array([[1.0, -0.5], [0.0, 1.0]]),
        "solucao": r"$M = \begin{bmatrix} 1 & -0{,}5 \\ 0 & 1 \end{bmatrix}$, ou seja, $x' = x - 0{,}5\,y$ e $y' = y$.",
    },
    {
        "titulo": "3. Inclinar na vertical, para cima",
        "texto": "Inclina a figura na vertical, de modo que cada vértice suba metade da sua distância ao eixo $Oy$ ($x$).",
        "alvo": np.array([[1, 4, 2], [1.5, 3, 4]], dtype=float),
        "M_sol": np.array([[1.0, 0.0], [0.5, 1.0]]),
        "solucao": r"$M = \begin{bmatrix} 1 & 0 \\ 0{,}5 & 1 \end{bmatrix}$, ou seja, $x' = x$ e $y' = y + 0{,}5\,x$.",
    },
    {
        "titulo": "4. Inclinar na vertical, para baixo",
        "texto": "Inclina a figura na vertical, de modo que cada vértice desça tanto quanto a sua distância ao eixo $Oy$ ($x$).",
        "alvo": np.array([[1, 4, 2], [0, -3, 1]], dtype=float),
        "M_sol": np.array([[1.0, 0.0], [-1.0, 1.0]]),
        "solucao": r"$M = \begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix}$, ou seja, $x' = x$ e $y' = y - x$.",
    },
    {
        "titulo": "5. Pontos fixos",
        "texto": (
            "Há algum cisalhamento (com $p = s = 1$ e só um de $q$ e $r$ diferente de zero) que **mantenha o vértice $C$** "
            "no mesmo sítio? E algum que mantenha $A$ ou $B$? Que pontos do plano ficam sempre fixos?"
        ),
        "alvo": None,
        "solucao": (
            "Nenhum: num cisalhamento horizontal só ficam fixos os pontos com $y = 0$ (o eixo $Ox$) e, num vertical, "
            "os pontos com $x = 0$ (o eixo $Oy$). Os vértices $A$, $B$ e $C$ não estão nesses eixos."
        ),
    },
]

DESAFIOS_RO = [
    {
        "titulo": "1. Um quarto de volta",
        "texto": "Roda a figura um quarto de volta em torno da origem, no sentido contrário ao dos ponteiros do relógio. *(Que fração de $\\pi$ corresponde a um quarto de volta?)*",
        "alvo": np.array([[-1, -1, -3], [1, 4, 2]], dtype=float),
        "c_sol": 0.5,
        "solucao": r"$c = 0{,}5$, isto é, $\alpha = \pi/2$ rad, que dá $M = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$.",
    },
    {
        "titulo": "2. Meia volta",
        "texto": "Roda a figura meia volta em torno da origem.",
        "alvo": -X,
        "c_sol": 1.0,
        "solucao": r"$c = 1$, isto é, $\alpha = \pi$ rad, que dá $M = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}$ (também $c = -1$).",
    },
    {
        "titulo": "3. Três quartos de volta",
        "texto": "Roda a figura três quartos de volta, no sentido contrário ao dos ponteiros do relógio. Há **outro** valor de $c$ (negativo) que dá a mesma imagem. Qual?",
        "alvo": np.array([[1, 1, 3], [-1, -4, -2]], dtype=float),
        "c_sol": 1.5,
        "solucao": r"$c = 1{,}5$ ou $c = -0{,}5$; a matriz é $M = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$.",
    },
    {
        "titulo": "4. Rotação de 45°",
        "texto": "Roda a figura 45° em torno da origem, no sentido contrário ao dos ponteiros do relógio. *(Quantos radianos são 45°, como múltiplo de $\\pi$?)*",
        "alvo": matriz_rotacao(np.pi / 4) @ X,
        "c_sol": 0.25,
        "solucao": r"$c = 0{,}25$, isto é, $\alpha = \pi/4$ rad.",
    },
    {
        "titulo": "5. Desafio aberto",
        "texto": (
            "Descobre **dois valores diferentes de $c$** que dão **exatamente a mesma imagem**. "
            "Que relação existe entre eles? E entre $c$ e $-c$?"
        ),
        "alvo": None,
        "solucao": r"Valores de $c$ que diferem de um número inteiro par (por exemplo $0{,}25$ e $2{,}25$) dão a mesma rotação; $c$ e $-c$ dão rotações inversas.",
    },
]


# --------------------------------------------------------------------------
# Blocos reutilizáveis dos módulos
# --------------------------------------------------------------------------
REGRA_MD = (
    "Cada vértice $\\mathbf{x}_i$ da matriz $X$ é transformado em $\\mathbf{x}_i' = M\\,\\mathbf{x}_i$, "
    "com $M = \\begin{bmatrix} p & q \\\\ r & s \\end{bmatrix}$; isto é, $X' = M\\,X$."
)


def secao_desafios(P, desafios, modo_prof, entrada=None, rotacao=False, circulos=False):
    """Seletor de desafios, entrada da matriz, verificação do alvo e medições."""
    escolha = st.selectbox("Desafio", [d["titulo"] for d in desafios], key=f"{P}_desafio")
    d = next(x for x in desafios if x["titulo"] == escolha)
    idx = desafios.index(d)
    st.markdown(d["texto"])
    ce, cg = st.columns([1, 1.3])
    with ce:
        M = entrada(f"{P}_d{idx}") if entrada else entrada_matriz(f"{P}_d{idx}")
    with cg:
        X1 = M @ X
        mostrar(desenhar([X, X1], ["original", "imagem"], alvo=d["alvo"], circulos=circulos))
        if d["alvo"] is not None:
            if np.allclose(X1, d["alvo"]):
                st.success("✅ Coincide com o alvo!")
            else:
                st.error("❌ Ainda não coincide com o alvo.")
    mostrar_medidas(X, X1, rotacao=rotacao)
    if modo_prof:
        st.info("**Solução:** " + d["solucao"])
    return idx


def caderno(P, titulo, obs_linhas, desafios, sintese_perguntas, n_conj=2):
    """Caderno de bordo do módulo, com exportação em Markdown."""
    st.subheader("📓 Caderno de bordo do módulo")
    st.caption("O caderno fica guardado enquanto a página não for recarregada. Descarrega-o no fim.")
    grupo = st.text_input("Nomes do grupo", key=f"{P}_cb_grupo")

    st.markdown("**Exploração**")
    df1 = pd.DataFrame({
        "Situação": obs_linhas,
        "O que observei": [""] * len(obs_linhas),
        "Que conjetura posso fazer?": [""] * len(obs_linhas),
    })
    ed1 = st.data_editor(df1, hide_index=True, use_container_width=True, key=f"{P}_cb1", disabled=["Situação"])

    st.markdown("**Desafios**")
    df2 = pd.DataFrame({
        "Desafio": [d["titulo"] for d in desafios],
        "M que usei": [""] * len(desafios),
        "Resultou? O que tentei (e não funcionou)": [""] * len(desafios),
    })
    ed2 = st.data_editor(df2, hide_index=True, use_container_width=True, key=f"{P}_cb2", disabled=["Desafio"])

    st.markdown("**Conjeturas**")
    conjeturas = []
    for i in range(1, n_conj + 1):
        with st.expander(f"Conjetura {i}", expanded=(i == 1)):
            c = st.text_area("Quando [condição], a figura [efeito].", key=f"{P}_cb_c{i}")
            t = st.text_area("Como a testei (valores usados)", key=f"{P}_cb_t{i}")
            res = st.radio("Resistiu?", ["—", "Sim", "Não (contraexemplo)"], horizontal=True, key=f"{P}_cb_r{i}")
            conjeturas.append((c, t, res))

    st.markdown("**Justificação**")
    just = st.text_area("Conjetura escolhida e justificação com cálculo", key=f"{P}_cb_just", height=140)

    st.markdown("**Síntese do módulo**")
    df5 = pd.DataFrame({"Pergunta": sintese_perguntas, "A minha resposta": [""] * len(sintese_perguntas)})
    ed5 = st.data_editor(df5, hide_index=True, use_container_width=True, key=f"{P}_cb5", disabled=["Pergunta"])

    md = [f"# Caderno de bordo — {titulo}", "", f"**Grupo:** {grupo}", "",
          "## Exploração", "", tabela_md(ed1), "",
          "## Desafios", "", tabela_md(ed2), "",
          "## Conjeturas", ""]
    for i, (c, t, res) in enumerate(conjeturas, start=1):
        md += [f"**Conjetura {i}:** {c}", "", f"- Teste: {t}", f"- Resistiu? {res}", ""]
    md += ["## Justificação", "", just, "", "## Síntese do módulo", "", tabela_md(ed5), ""]
    st.download_button("⬇️ Descarregar o caderno deste módulo (Markdown)", "\n".join(md),
                       file_name=f"caderno_{P}.md", mime="text/markdown", key=f"{P}_cb_dl")


# --------------------------------------------------------------------------
# Módulo A — Variações de tamanho (scaling)
# --------------------------------------------------------------------------
@st.fragment
def sc_explorar():
    P = "sc"
    st.subheader("1 · Explorar")
    st.markdown(
        "Começa com $M$ igual à identidade. Altera **só $p$**; depois **só $s$**; depois $p$ e $s$ **ao mesmo tempo**. "
        "O que acontece à figura? E se mexeres também em $q$ ou $r$? Regista o que observas no caderno de bordo do módulo."
    )
    cs, cg = st.columns([1, 1.3])
    with cs:
        M = sliders_matriz(f"{P}_e")
        st.latex(rf"M = {latex_mat(M)}")
    with cg:
        mostrar(desenhar([X, M @ X], ["original", "imagem"]))
    mostrar_medidas(X, M @ X)


@st.fragment
def sc_desafios(modo_prof):
    P = "sc"
    st.divider()
    st.subheader("2 · Caça às transformações")
    st.markdown(
        "Em cada desafio, **descobre** a matriz $M$ que produz o efeito pedido. Se houver um **alvo** (tracejado), "
        "tenta fazer coincidir a imagem com ele. Regista o que tentaste, inclusive o que **não** funcionou."
    )
    secao_desafios(P, DESAFIOS_SC, modo_prof)


@st.fragment
def sc_conjeturar():
    P = "sc"
    st.divider()
    st.subheader("3 · Conjeturar e testar")
    st.markdown(
        "Formula **por escrito** pelo menos duas conjeturas, no formato *\"Quando [condição sobre os números], a figura [efeito].\"* "
        "Sugestões:\n"
        "- Quando é que a figura mantém a **forma** (só muda de tamanho)? E quando é deformada?\n"
        "- Que relação existe entre $p$, $s$ e a **razão das áreas**? (Experimenta valores positivos e negativos.)\n"
        "- O que fazem os valores **negativos**? A figura continua a ser \"só maior ou menor\"?\n"
        "- Que pontos ficam sempre no mesmo sítio?\n\n"
        "Para cada conjetura, escreve **como a testaste** e **se resistiu**. Procura um contraexemplo antes de a dares por boa."
    )
    st.markdown("**Laboratório de conjeturas**")
    ce, cg = st.columns([1, 1.3])
    with ce:
        Ml = entrada_matriz(f"{P}_lab")
    with cg:
        mostrar(desenhar([X, Ml @ X], ["original", "imagem"]))
    mostrar_medidas(X, Ml @ X)

    st.markdown("**Compor dois scalings:** aplica-se $M_1$ e depois $M_2$. O que observas no produto $M_2\\,M_1$?")
    cc, cd = st.columns([1, 1.3])
    with cc:
        k1 = st.number_input("k₁ (p de M₁)", value=2.0, step=0.5, key=f"{P}_k1")
        l1 = st.number_input("l₁ (s de M₁)", value=1.0, step=0.5, key=f"{P}_l1")
        k2 = st.number_input("k₂ (p de M₂)", value=1.0, step=0.5, key=f"{P}_k2")
        l2 = st.number_input("l₂ (s de M₂)", value=3.0, step=0.5, key=f"{P}_l2")
    with cd:
        mostrar_composicao(np.diag([k1, l1]), np.diag([k2, l2]))


@st.fragment
def sc_justificar(modo_prof):
    P = "sc"
    st.divider()
    st.subheader("4 · Justificar")
    st.markdown(
        "Escolhe **uma** conjetura que tenha resistido e justifica-a com cálculo. Por exemplo: porque é que a razão das áreas é $|p\\,s|$ "
        "quando $q = r = 0$? Podes usar o que vês abaixo: as colunas de $M$, o produto $M\\,X$ e o determinante."
    )
    Mj = entrada_matriz(f"{P}_just", ((2.0, 0.0), (0.0, 3.0)))
    justificar_comum(Mj)
    if modo_prof:
        st.info(
            "**Conclusões esperadas:**\n"
            "- Com $M = \\begin{bmatrix} k & 0 \\\\ 0 & l \\end{bmatrix}$: $x' = k\\,x$ e $y' = l\\,y$; a razão das áreas é $|k\\,l|$ (o determinante em valor absoluto).\n"
            "- A **forma** é mantida quando $|k| = |l|$; se $k \\neq l$ a figura é deformada (os ângulos mudam).\n"
            "- Fatores negativos refletem a figura (um fator) ou dão meia volta (dois fatores negativos).\n"
            "- A origem fica sempre fixa; se $k = 1$, ficam fixos todos os pontos de $Ox$.\n"
            "- Compor scalings multiplica os fatores: $M_2\\,M_1 = \\mathrm{diag}(k_1 k_2,\\, l_1 l_2)$."
        )


@st.fragment
def sc_caderno():
    P = "sc"
    st.divider()
    caderno(
        P, "Módulo A · Variações de tamanho (scaling)",
        ["só p", "só s", "p = s", "p ou s negativos", "q ou r diferentes de 0"],
        DESAFIOS_SC,
        ["Forma geral da matriz de scaling",
         "Efeito de cada elemento da diagonal",
         "Relação entre os fatores e a razão das áreas",
         "Efeito de fatores negativos",
         "O que se conserva e o que não se conserva"],
    )


def modulo_scaling(modo_prof):
    st.header("Módulo A — Variações de tamanho (scaling)")
    st.markdown(REGRA_MD)
    st.markdown("Neste módulo vais descobrir que matrizes **ampliam, reduzem ou deformam** a figura em relação à origem.")
    sc_explorar()
    sc_desafios(modo_prof)
    sc_conjeturar()
    sc_justificar(modo_prof)
    sc_caderno()


# --------------------------------------------------------------------------
# Módulo B — Cisalhamentos (shearing)
# --------------------------------------------------------------------------
@st.fragment
def sh_explorar():
    P = "sh"
    st.subheader("1 · Explorar")
    st.markdown(
        "Começa com $M$ igual à identidade. Altera **só $q$**; depois volta a $q = 0$ e altera **só $r$**. "
        "Compara. Que coordenada dos vértices muda em cada caso? Que pontos não se mexem? "
        "Regista o que observas no caderno de bordo do módulo."
    )
    cs, cg = st.columns([1, 1.3])
    with cs:
        M = sliders_matriz(f"{P}_e")
        st.latex(rf"M = {latex_mat(M)}")
    with cg:
        mostrar(desenhar([X, M @ X], ["original", "imagem"]))
    mostrar_medidas(X, M @ X)


@st.fragment
def sh_desafios(modo_prof):
    P = "sh"
    st.divider()
    st.subheader("2 · Caça às transformações")
    st.markdown(
        "Em cada desafio, **descobre** a matriz $M$ que produz o efeito pedido. Se houver um **alvo** (tracejado), "
        "tenta fazer coincidir a imagem com ele. Regista o que tentaste, inclusive o que **não** funcionou."
    )
    secao_desafios(P, DESAFIOS_SH, modo_prof)


@st.fragment
def sh_conjeturar():
    P = "sh"
    st.divider()
    st.subheader("3 · Conjeturar e testar")
    st.markdown(
        "Formula **por escrito** pelo menos duas conjeturas, no formato *\"Quando [condição sobre os números], a figura [efeito].\"* "
        "Sugestões:\n"
        "- Que coordenada dos vértices se altera e qual se mantém? De que depende a alteração?\n"
        "- Que pontos ficam fixos? (Olha para os vértices e para os eixos.)\n"
        "- O que acontece à **área**? E aos **lados** e aos **ângulos**?\n"
        "- O que acontece se aplicares o cisalhamento com $q$ e depois com $-q$?\n\n"
        "Para cada conjetura, escreve **como a testaste** e **se resistiu**. Procura um contraexemplo antes de a dares por boa."
    )
    st.markdown("**Laboratório de conjeturas**")
    ce, cg = st.columns([1, 1.3])
    with ce:
        Ml = entrada_matriz(f"{P}_lab")
    with cg:
        mostrar(desenhar([X, Ml @ X], ["original", "imagem"]))
    mostrar_medidas(X, Ml @ X)

    st.markdown("**Compor dois cisalhamentos horizontais:** aplica-se $M_1$ e depois $M_2$. O que observas no produto $M_2\\,M_1$?")
    cc, cd = st.columns([1, 1.3])
    with cc:
        a = st.number_input("q de M₁", value=1.0, step=0.5, key=f"{P}_a")
        b = st.number_input("q de M₂", value=-1.0, step=0.5, key=f"{P}_b")
    with cd:
        mostrar_composicao(np.array([[1.0, a], [0.0, 1.0]]), np.array([[1.0, b], [0.0, 1.0]]))


@st.fragment
def sh_justificar(modo_prof):
    P = "sh"
    st.divider()
    st.subheader("4 · Justificar")
    st.markdown(
        "Escolhe **uma** conjetura que tenha resistido e justifica-a com cálculo. Por exemplo: porque é que a **área se conserva**? "
        "Compara a razão das áreas com o determinante $ps - qr$ e observa o quadrado unitário e a sua imagem."
    )
    Mj = entrada_matriz(f"{P}_just", ((1.0, 1.0), (0.0, 1.0)))
    justificar_comum(Mj)
    if modo_prof:
        st.info(
            "**Conclusões esperadas:**\n"
            "- Com $M = \\begin{bmatrix} 1 & q \\\\ 0 & 1 \\end{bmatrix}$: $x' = x + q\\,y$ e $y' = y$ (horizontal). "
            "Com $M = \\begin{bmatrix} 1 & 0 \\\\ r & 1 \\end{bmatrix}$: $x' = x$ e $y' = y + r\\,x$ (vertical).\n"
            "- A **área conserva-se** ($\\det M = 1$): o quadrado unitário passa a um paralelogramo com a mesma base e a mesma altura.\n"
            "- Os **lados** e os **ângulos** não se conservam.\n"
            "- Ficam fixos os pontos do eixo $Ox$ (horizontal) ou do eixo $Oy$ (vertical).\n"
            "- Compor cisalhamentos horizontais soma os $q$: o de $-q$ \"desfaz\" o de $q$."
        )


@st.fragment
def sh_caderno():
    P = "sh"
    st.divider()
    caderno(
        P, "Módulo B · Cisalhamentos (shearing)",
        ["só q (p = s = 1, r = 0)", "só r (p = s = 1, q = 0)", "q negativo", "valores grandes de q ou r"],
        DESAFIOS_SH,
        ["Forma geral do cisalhamento horizontal",
         "Forma geral do cisalhamento vertical",
         "Pontos que ficam fixos",
         "Efeito na área",
         "O que se conserva e o que não se conserva"],
    )


def modulo_shearing(modo_prof):
    st.header("Módulo B — Cisalhamentos (shearing)")
    st.markdown(REGRA_MD)
    st.markdown("Neste módulo vais descobrir que matrizes **inclinam** a figura, \"deslizando-a\" numa direção.")
    sh_explorar()
    sh_desafios(modo_prof)
    sh_conjeturar()
    sh_justificar(modo_prof)
    sh_caderno()


# --------------------------------------------------------------------------
# Módulo C — Rotações
# --------------------------------------------------------------------------
@st.fragment
def ro_explorar():
    P = "ro"
    st.subheader("1 · Explorar")
    st.markdown(
        "Move $c$. O que **se mantém** na figura (lados, ângulos, área)? Em torno de que ponto roda? "
        "Que trajetória faz cada vértice (as circunferências a tracejado)? O que acontece com $c = 1$ e com $c = 2$? E com $c$ negativo? "
        "Regista o que observas no caderno de bordo do módulo."
    )
    cs, cg = st.columns([1, 1.3])
    with cs:
        M = entrada_rotacao(f"{P}_e", slider=True)
    with cg:
        mostrar(desenhar([X, M @ X], ["original", "imagem"], circulos=True))
    mostrar_medidas(X, M @ X, rotacao=True)


@st.fragment
def ro_desafios(modo_prof):
    P = "ro"
    st.divider()
    st.subheader("2 · Caça às transformações")
    st.markdown(
        "Em cada desafio, **descobre** o valor de $c$ que produz o efeito pedido. Se houver um **alvo** (tracejado), "
        "tenta fazer coincidir a imagem com ele. Regista o que tentaste, inclusive o que **não** funcionou."
    )
    secao_desafios(P, DESAFIOS_RO, modo_prof, entrada=entrada_rotacao, rotacao=True, circulos=True)


@st.fragment
def ro_conjeturar():
    P = "ro"
    st.divider()
    st.subheader("3 · Conjeturar e testar")
    st.markdown(
        "Formula **por escrito** pelo menos duas conjeturas, no formato *\"Quando [condição sobre os números], a figura [efeito].\"* "
        "Sugestões:\n"
        "- O que se conserva: **lados**, **ângulos**, **área**?\n"
        "- O que acontece a $M$ com $\\alpha$ e $-\\alpha$ (por exemplo $c = 0{,}25$ e $c = -0{,}25$)? Que rotação \"desfaz\" a de ângulo $\\alpha$?\n"
        "- O que acontece com $\\alpha$ e $\\alpha + 2\\pi$ (isto é, $c$ e $c + 2$)?\n"
        "- Que relação existe entre os ângulos de duas rotações compostas e o ângulo da rotação resultante?\n\n"
        "Para cada conjetura, escreve **como a testaste** e **se resistiu**. Procura um contraexemplo antes de a dares por boa."
    )
    st.markdown("**Laboratório de conjeturas**")
    ce, cg = st.columns([1, 1.3])
    with ce:
        Ml = entrada_rotacao(f"{P}_lab")
    with cg:
        mostrar(desenhar([X, Ml @ X], ["original", "imagem"], circulos=True))
    mostrar_medidas(X, Ml @ X, rotacao=True)

    st.markdown("**Compor duas rotações:** aplica-se $M_1 = M(c_1\\pi)$ e depois $M_2 = M(c_2\\pi)$. Compara o produto com $M((c_1 + c_2)\\pi)$.")
    cc, cd = st.columns([1, 1.3])
    with cc:
        c1 = st.number_input("c₁", value=0.25, step=0.05, format="%.3f", key=f"{P}_comp_c1")
        c2 = st.number_input("c₂", value=0.5, step=0.05, format="%.3f", key=f"{P}_comp_c2")
    with cd:
        Pr = mostrar_composicao(matriz_rotacao(c1 * np.pi), matriz_rotacao(c2 * np.pi), circulos=True)
        Ms = matriz_rotacao((c1 + c2) * np.pi)
        st.latex(rf"M\big((c_1 + c_2)\pi\big) = {latex_mat(Ms)}")
        st.write("**As duas matrizes são iguais?**", "Sim ✅" if np.allclose(Pr, Ms) else "Não ❌")


@st.fragment
def ro_justificar(modo_prof):
    P = "ro"
    st.divider()
    st.subheader("4 · Justificar")
    st.markdown(
        "Escolhe **uma** conjetura que tenha resistido e justifica-a com cálculo. Por exemplo: porque é que as colunas de $M$ são "
        "$(\\cos\\alpha,\\ \\sin\\alpha)$ e $(-\\sin\\alpha,\\ \\cos\\alpha)$? *Sugestão:* onde vão $\\mathbf{e}_1$ e $\\mathbf{e}_2$ no círculo trigonométrico? "
        "E porque é que o determinante é sempre $1$?"
    )
    Mj = entrada_rotacao(f"{P}_just", c0=0.25)
    justificar_comum(Mj)
    if modo_prof:
        st.info(
            "**Conclusões esperadas:**\n"
            "- A imagem de $\\mathbf{e}_1 = (1,0)$ é o ponto do círculo trigonométrico $(\\cos\\alpha,\\ \\sin\\alpha)$; "
            "a de $\\mathbf{e}_2 = (0,1)$ é $(-\\sin\\alpha,\\ \\cos\\alpha)$ (rodada mais 90°). São as colunas de $M(\\alpha)$.\n"
            "- A rotação conserva **lados, ângulos e área**: $\\det M = \\cos^2\\alpha + \\sin^2\\alpha = 1$.\n"
            "- A origem é o único ponto fixo (para $c$ não inteiro par). $M(\\alpha + 2\\pi) = M(\\alpha)$.\n"
            "- Compor rotações soma os ângulos: $M(\\beta)\\,M(\\alpha) = M(\\alpha + \\beta)$; a inversa de $M(\\alpha)$ é $M(-\\alpha)$."
        )


@st.fragment
def ro_caderno():
    P = "ro"
    st.divider()
    caderno(
        P, "Módulo C · Rotações",
        ["c > 0", "c < 0", "c = 0,5", "c = 1", "c = 2"],
        DESAFIOS_RO,
        ["Matriz da rotação de ângulo α em torno da origem",
         "Relação das colunas de M com o círculo trigonométrico",
         "O que se conserva",
         "Rotação que desfaz a de ângulo α; composição de rotações"],
    )


def modulo_rotacoes(modo_prof):
    st.header("Módulo C — Rotações")
    st.markdown(REGRA_MD)
    st.markdown(
        "Neste módulo vais descobrir a matriz que **roda** a figura em torno da origem. "
        "O ângulo de rotação, em radianos, é $\\alpha = c\\,\\pi$; o que escolhes é o valor de $c$ "
        "(positivo = sentido contrário ao dos ponteiros do relógio)."
    )
    ro_explorar()
    ro_desafios(modo_prof)
    ro_conjeturar()
    ro_justificar(modo_prof)
    ro_caderno()


# --------------------------------------------------------------------------
# Módulo D — Exercício: levar o vetor x ao alvo por uma rotação (em construção)
# --------------------------------------------------------------------------
XV = np.array([2.0, 1.0])  # o vetor x, fixo
ALVO_V = matriz_rotacao(np.pi / 2) @ XV  # alvo: x rodado de π/2, isto é (-1, 2)

# opções de cada entrada de M: nome -> (função do valor v, LaTeX do resultado)
# "=" : a entrada é o próprio valor v;  "=sin", "=cos", ...: a entrada é sin(v·π), cos(v·π), ... (ângulo em radianos, v·π)
OPCOES_M = {
    "=": (lambda v: v, lambda v: _f(v)),
    "=-sin": (lambda v: -np.sin(v * np.pi), lambda v: rf"-\sin({_f(v)}\pi)"),
    "=sin": (lambda v: np.sin(v * np.pi), lambda v: rf"\sin({_f(v)}\pi)"),
    "=-cos": (lambda v: -np.cos(v * np.pi), lambda v: rf"-\cos({_f(v)}\pi)"),
    "=cos": (lambda v: np.cos(v * np.pi), lambda v: rf"\cos({_f(v)}\pi)"),
}


def latex_pmat(A):
    """Como latex_mat, mas com parênteses curvos."""
    return latex_mat(A).replace("bmatrix", "pmatrix")


def desenhar_vetor(x, x1, alvo=None):
    """Setas de x e de x' = M x a partir da origem; o vetor alvo, a tracejado."""
    pts = [x, x1] + ([alvo] if alvo is not None else [])
    lim = max(3.0, float(np.ceil(np.abs(np.concatenate(pts)).max())) + 1)
    fig = Figure(figsize=(5.5, 5.5))
    ax = fig.subplots()
    if alvo is not None:
        ax.annotate("", xy=alvo, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="k", lw=2, ls="--"))
        ax.annotate("alvo", alvo, textcoords="offset points", xytext=(8, -14), color="k", fontsize=12)
    for v, cor, nome in ((x, "tab:blue", r"$\mathbf{x}$"), (x1, "tab:red", r"$\mathbf{x}'$")):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=cor, lw=2.5))
        ax.plot(*v, "o", color=cor)
        ax.annotate(f"{nome} = ({_f(v[0])}, {_f(v[1])})", v, textcoords="offset points",
                    xytext=(8, 8), color=cor, fontsize=12)
    _eixos(ax, lim)
    return fig


def _parenteses(col, simbolo):
    col.markdown(f"<div style='font-size:5.6rem;line-height:1;font-weight:200;text-align:center'>{simbolo}</div>",
                 unsafe_allow_html=True)


@st.fragment
def dv_produto(modo_prof):
    st.subheader("Leva o vetor $\\mathbf{x}$ até ao alvo")
    st.markdown(
        "O vetor $\\mathbf{x} = (2,1)$ é fixo. O **alvo** (a tracejado) é a imagem de $\\mathbf{x}$ por uma **rotação** em torno da origem. "
        "Monta a matriz $M$ que faz $\\mathbf{x}' = M\\,\\mathbf{x}$ coincidir com o alvo. "
        "Cada entrada de $M$ tem um menu e, à direita, um campo com um valor $v$. "
        "Com **=** a entrada é o próprio $v$; com **=sin**, **=cos** (ou com sinal −) a entrada é "
        "$\\sin(v\\,\\pi)$, $\\cos(v\\,\\pi)$ (ou o simétrico): o ângulo é $v\\,\\pi$ radianos."
    )
    with st.expander("💡 Dica: usa os números complexos"):
        st.markdown(
            "Identifica o vetor $(x,y)$ com o número complexo $z = x + iy$. "
            "Assim, $\\mathbf{x}$ corresponde a $2 + i$, e o alvo corresponde a $-1 + 2i$.\n\n"
            "Rodar um vetor de um ângulo $\\theta$ em torno da origem é **multiplicar** o seu número complexo por "
            "$w = \\cos\\theta + i\\sin\\theta$.\n\n"
            "1. Descobre o número complexo $w$ tal que $w\\,(2+i) = -1+2i$.\n"
            "2. Lê em $w$ os valores de $\\cos\\theta$ (parte real) e de $\\sin\\theta$ (parte imaginária); "
            "daí tiras o ângulo $\\theta$.\n"
            "3. Multiplica $(\\cos\\theta + i\\sin\\theta)(x + iy)$ e separa a parte real da parte imaginária. "
            "Escreve cada uma na forma $\\square\\,x + \\square\\,y$: os coeficientes são as **linhas** de $M$.\n"
            "4. Escolhe, para cada entrada, o menu e o valor $v$ (lembra-te de que o ângulo é $v\\,\\pi$)."
        )
    if modo_prof:
        st.info(
            "**Solução:** $w = \\dfrac{-1+2i}{2+i} = i$, logo $\\cos\\theta = 0$, $\\sin\\theta = 1$ e $\\theta = \\pi/2$. "
            "$M = \\begin{pmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta \\end{pmatrix}$: "
            "menus `=cos`, `=-sin`, `=sin`, `=cos`, todos com $v = 0{,}5$."
        )
    esq, dir_ = st.columns([2.3, 1])
    with esq:
        c = st.columns([0.3, 0.7, 0.7, 0.7, 0.7, 0.3, 0.15, 0.6])
        _parenteses(c[0], "("); _parenteses(c[5], ")")
        nomes = list(OPCOES_M)
        padrao = {(0, 0): 1.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 1.0}
        esc, val = [[None, None], [None, None]], [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                esc[i][j] = c[1 + 2 * j].selectbox(f"m{i + 1}{j + 1}", nomes, key=f"dv_f{i}{j}",
                                                   label_visibility="collapsed")
                val[i][j] = c[2 + 2 * j].number_input(f"v{i + 1}{j + 1}", value=padrao[(i, j)], step=0.25,
                                                      key=f"dv_v{i}{j}", label_visibility="collapsed")
        c[7].latex(latex_pmat(XV.reshape(-1, 1)))
        M = np.array([[OPCOES_M[esc[i][j]][0](val[i][j]) for j in range(2)] for i in range(2)])
        M_simb = (r"\begin{pmatrix} " + r" \\ ".join(
            " & ".join(OPCOES_M[esc[i][j]][1](val[i][j]) for j in range(2)) for i in range(2)) + r" \end{pmatrix}")
        x1 = M @ XV
        st.markdown("$M$ (à esquerda) multiplica $\\mathbf{x}$ (à direita).")
        st.latex(rf"M = {M_simb} = {latex_pmat(M)}")
        st.latex(rf"\mathbf{{x}}' = M\,\mathbf{{x}} = {latex_pmat(M)}{latex_pmat(XV.reshape(-1, 1))}"
                 rf" = {latex_pmat(x1.reshape(-1, 1))}")
        if np.allclose(x1, ALVO_V):
            st.success("✅ $\\mathbf{x}'$ coincide com o alvo!")
        else:
            st.error("❌ $\\mathbf{x}'$ ainda não coincide com o alvo.")
    with dir_:
        mostrar(desenhar_vetor(XV, x1, ALVO_V))


def modulo_exercicio(modo_prof):
    st.header("Módulo D — Exercício")
    dv_produto(modo_prof)


# --------------------------------------------------------------------------
# Escalonamento — rascunho de uma tarefa curta, com o objetivo de cada parte indicado
# --------------------------------------------------------------------------
def nota_objetivo(conhecimento, capacidade, evidencia):
    """Caixa de rascunho: o que se pretende desenvolver na parte em causa."""
    st.info(
        "🎯 **Rascunho · o que se pretende desenvolver nesta parte**\n\n"
        f"- **Conhecimento:** {conhecimento}\n"
        f"- **Capacidade:** {capacidade}\n"
        f"- **Evidência:** {evidencia}"
    )


def desenhar_escalonamento(k, l, pontos=True, seg_h=False, seg_v=False, quadrado=False):
    """Triângulo X e a sua imagem por diag(k, l), com ajudas visuais opcionais:
    deslocamento de cada ponto, base AB (segmento horizontal) e altura CD (segmento vertical) com as suas imagens,
    quadrado unitário e a sua imagem."""
    M = np.array([[k, 0.0], [0.0, l]])
    X1 = M @ X
    D = np.array([X[0, 2], X[1, 0]])  # pé da altura traçada de C sobre a base AB (horizontal)
    D1 = M @ D
    Q = np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float)
    ext = [np.abs(X).max(), np.abs(X1).max(), 5.0]
    if quadrado:
        ext.append(max(k, l))
    lim = float(np.ceil(max(ext))) + 1
    fig = Figure(figsize=(5.5, 5.5))
    ax = fig.subplots()
    if quadrado:
        for P, cor, nome in ((Q, "tab:blue", "quadrado unitário"), (M @ Q, "tab:red", "imagem do quadrado")):
            ax.fill(P[0], P[1], alpha=0.12, color=cor, zorder=0)
            ax.plot(np.append(P[0], P[0][0]), np.append(P[1], P[1][0]), "-", color=cor, lw=1.2, zorder=1, label=nome)
    if seg_h:
        for (p, q), cor, nome in (((X[:, 0], X[:, 1]), "tab:blue", "AB"), ((X1[:, 0], X1[:, 1]), "tab:red", "A′B′")):
            ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=cor, lw=5, alpha=0.55, solid_capstyle="butt", zorder=3)
            ax.annotate(nome, (p + q) / 2, textcoords="offset points", xytext=(0, -16), ha="center",
                        color=cor, fontsize=10, fontweight="bold")
    if seg_v:
        for (p, q), cor, nome, d in (((D, X[:, 2]), "tab:blue", "CD", "D"), ((D1, X1[:, 2]), "tab:red", "C′D′", "D′")):
            ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=cor, lw=5, alpha=0.55, solid_capstyle="butt", zorder=3)
            ax.plot(*p, "s", color=cor, ms=5, zorder=4)
            ax.annotate(d, p, textcoords="offset points", xytext=(6, -14), color=cor, fontsize=10)
            ax.annotate(nome, (p + q) / 2, textcoords="offset points", xytext=(8, 0), color=cor,
                        fontsize=10, fontweight="bold")
    for P, cor, nome, suf in ((X, "tab:blue", "original", ""), (X1, "tab:red", "imagem", "′")):
        ax.fill(np.append(P[0], P[0][0]), np.append(P[1], P[1][0]), alpha=0.2, color=cor)
        ax.plot(np.append(P[0], P[0][0]), np.append(P[1], P[1][0]), "o-", color=cor, label=nome)
        for j in range(3):
            ax.annotate(f"{chr(65 + j)}{suf}", (P[0][j], P[1][j]), textcoords="offset points",
                        xytext=(7, 7), color=cor, fontsize=11, fontweight="bold")
    if pontos:
        for j in range(3):
            if not np.allclose(X[:, j], X1[:, j]):
                ax.annotate("", xy=X1[:, j], xytext=X[:, j],
                            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2, ls="--", shrinkA=3, shrinkB=3))
    _eixos(ax, lim)
    ax.legend(loc="upper left")
    return fig


@st.fragment
def es_explorar(modo_prof):
    st.subheader("Parte 1 · Explorar e conjeturar")
    nota_objetivo(
        "Uma variação de tamanho, $\\mathrm{diag}(k,l)$, leva $(x,y)$ a $(kx,\\,ly)$; um segmento horizontal continua horizontal, com o comprimento multiplicado por $k$, "
        "e um segmento vertical continua vertical, com o comprimento multiplicado por $l$; a área fica multiplicada por $k\\,l$.",
        "Formular **conjeturas** a partir da observação (gráfico e valores numéricos) sobre o que a transformação faz a "
        "(a) cada ponto, (b) cada segmento horizontal e vertical e (c) à área, e exprimi-las em linguagem matemática.",
        "O aluno escreve uma regra para as coordenadas, diz que os segmentos mantêm a direção, indica como mudam os comprimentos, e relaciona as áreas com $k$ e $l$.",
    )
    st.markdown(
        "Move $k$ e $l$ e liga as ajudas no gráfico (uma de cada vez, para cada pergunta). "
        "Experimenta valores **iguais** e **diferentes**, maiores e menores do que 1."
    )
    c1, c2 = st.columns([1, 1.4])
    with c1:
        k = st.slider("k", 0.25, 3.0, 1.0, 0.25, key="es_k1")
        l = st.slider("l", 0.25, 3.0, 1.0, 0.25, key="es_l1")
        M = np.array([[k, 0.0], [0.0, l]])
        st.latex(rf"M = {latex_mat(M)}")
        st.markdown("**Ajudas no gráfico**")
        pontos = st.checkbox("Deslocamento de cada ponto (A→A′, B→B′, C→C′)", value=True, key="es_v_pontos")
        seg_h = st.checkbox("Segmento horizontal (a base AB) e a sua imagem", value=False, key="es_v_h")
        seg_v = st.checkbox("Segmento vertical (a altura CD) e a sua imagem", value=False, key="es_v_v")
        quadrado = st.checkbox("Quadrado unitário e a sua imagem", value=False, key="es_v_q")
        X1 = M @ X
        linhas = {"Ponto": [], "Original": [], "Imagem": []}
        for j in range(3):
            linhas["Ponto"].append(f"{chr(65 + j)} → {chr(65 + j)}′")
            linhas["Original"].append(f"({_f(X[0, j])}, {_f(X[1, j])})")
            linhas["Imagem"].append(f"({_f(X1[0, j])}, {_f(X1[1, j])})")
        st.markdown(tabela_md(pd.DataFrame(linhas)))
        if seg_h:
            st.metric("Comprimento da base AB (original → imagem)",
                      f"{np.linalg.norm(X[:, 1] - X[:, 0]):.2f} → {np.linalg.norm(X1[:, 1] - X1[:, 0]):.2f}")
        if seg_v:
            D = np.array([X[0, 2], X[1, 0]])
            st.metric("Comprimento da altura CD (original → imagem)",
                      f"{np.linalg.norm(X[:, 2] - D):.2f} → {np.linalg.norm(X1[:, 2] - M @ D):.2f}")
        st.metric("Área do triângulo (original → imagem)", f"{area(X):.2f} → {area(X1):.2f}")
    with c2:
        mostrar(desenhar_escalonamento(k, l, pontos, seg_h, seg_v, quadrado))
    st.markdown("**Conjetura (a) · Pontos.** Compara as coordenadas de cada ponto com as da sua imagem. "
                "Que fazes a $(x,y)$ para obter a imagem? Escreve a regra.")
    st.text_area("Pontos", key="es_t1a", height=70, label_visibility="collapsed")
    st.markdown("**Conjetura (b) · Segmentos.** Que acontece a um segmento horizontal (a base $AB$)? E a um segmento vertical "
                "(a altura $CD$)? Mantêm a direção? Como mudam os seus comprimentos?")
    st.text_area("Segmentos", key="es_t1b", height=70, label_visibility="collapsed")
    st.markdown("**Conjetura (c) · Área.** Como se relaciona a área da imagem com a da figura original? "
                "E com $k$ e $l$? *(O quadrado unitário pode ajudar.)*")
    st.text_area("Área", key="es_t1c", height=70, label_visibility="collapsed")
    if modo_prof:
        st.info(
            "**Conjeturas esperadas:** (a) $(x,y)\\mapsto(k\\,x,\\ l\\,y)$: a abcissa multiplica por $k$ e a ordenada por $l$. "
            "(b) Um segmento horizontal continua horizontal e o seu comprimento multiplica por $k$ (em módulo); um segmento vertical continua vertical e o seu comprimento multiplica por $l$. "
            "(c) A área fica multiplicada por $k\\,l$ (o quadrado unitário passa a um retângulo $k\\times l$)."
        )


def desenhar_alvo(X1, alvo):
    """Triângulo original (com as coordenadas de A, B, C), triângulo alvo (com as suas coordenadas) e a imagem atual."""
    lim = float(np.ceil(max(np.abs(X).max(), np.abs(alvo).max(), np.abs(X1).max(), 5.0))) + 1
    fig = Figure(figsize=(5.5, 5.5))
    ax = fig.subplots()
    estilos = ((X, "tab:blue", "-", "original", "", None),
               (X1, "tab:red", "-", "imagem de M", "′", None),
               (alvo, "k", "--", "alvo", "*", None))
    for P, cor, ls, nome, suf, desl in estilos:
        xs, ys = np.append(P[0], P[0][0]), np.append(P[1], P[1][0])
        if nome != "alvo":
            ax.fill(xs, ys, alpha=0.15, color=cor)
        ax.plot(xs, ys, ls, color=cor, marker="o", label=nome, lw=1.8 if nome != "imagem de M" else 1.4)
        if nome != "imagem de M":  # coordenadas de A, B, C e de A*, B*, C*
            for j in range(3):
                if nome == "original":  # etiquetas do original: em cima, à esquerda do ponto
                    desloc, ha = (-6, 6), "right"
                else:                   # etiquetas do alvo: em baixo, à direita do ponto
                    desloc, ha = (6, -15), "left"
                ax.annotate(f"{chr(65 + j)}{suf}({_f(P[0, j])}, {_f(P[1, j])})", (P[0, j], P[1, j]),
                            textcoords="offset points", xytext=desloc, ha=ha, color=cor, fontsize=10, fontweight="bold")
    _eixos(ax, lim)
    ax.legend(loc="upper left")
    return fig


@st.fragment
def es_inverso(modo_prof):
    st.subheader("Parte 2 · Descobrir $M$")
    nota_objetivo(
        "$M$ leva $(x,y)$ a $(kx,\\,ly)$, isto é, $M=\\begin{bmatrix}k&0\\\\0&l\\end{bmatrix}$; logo $k=x'/x$ e $l=y'/y$ em qualquer vértice.",
        "Usar a conjetura da Parte 1 como **ferramenta**: a partir das coordenadas de $A$ e do ponto correspondente do alvo, "
        "determinar as entradas de $M$ (incluindo as que são nulas), e verificar com os outros vértices.",
        "O aluno preenche $M$ sem tentativa e erro prolongada e justifica-a com um vértice.",
    )
    alvos = [
        ("Alvo 1", np.diag([2.0, 1.0]) @ X, "$M=\\begin{bmatrix}2&0\\\\0&1\\end{bmatrix}$"),
        ("Alvo 2", np.diag([0.5, 2.0]) @ X, "$M=\\begin{bmatrix}0{,}5&0\\\\0&2\\end{bmatrix}$"),
    ]
    esc = st.selectbox("Desafio", [a[0] for a in alvos], key="es_d3")
    idx = [a[0] for a in alvos].index(esc)
    alvo = alvos[idx][1]
    st.markdown(
        "O triângulo alvo (a tracejado) é a imagem de $ABC$ por uma matriz $M$ que tens de descobrir. "
        "O gráfico mostra as coordenadas de $A$, $B$, $C$ e dos pontos correspondentes do alvo, $A^*$, $B^*$, $C^*$. "
        "**Usa a regra que conjeturaste na Parte 1** para encontrar as quatro entradas de $M$. "
        "Quando as preencheres, a imagem aparece a vermelho."
    )
    c1, c2 = st.columns([1, 1.4])
    with c1:
        M = entrada_matriz(f"es_m3_{idx}")
        st.latex(rf"M = {latex_mat(M)}")
        X1 = M @ X
        if np.allclose(X1, alvo):
            st.success("✅ A imagem coincide com o alvo!")
        else:
            st.error("❌ A imagem ainda não coincide com o alvo.")
        if modo_prof:
            st.info("**Solução:** " + alvos[idx][2])
    with c2:
        mostrar(desenhar_alvo(X1, alvo))
    st.markdown(
        "Explica como podias ter descoberto as entradas de $M$ **olhando só para um vértice** e para o seu correspondente no alvo. "
        "Porque é que as entradas fora da diagonal são nulas?"
    )
    st.text_area("A tua explicação", key=f"es_t3_{idx}", height=80, label_visibility="collapsed")


def _passo(chave, pergunta, opcoes, certa, dica):
    """Passo de uma justificação guiada: escolha com resposta imediata. Devolve True se estiver certo."""
    st.markdown(pergunta)
    esc = st.radio("Escolhe", opcoes, index=None, key=chave, horizontal=True, label_visibility="collapsed")
    if esc is None:
        return False
    if esc == certa:
        st.success("✅ Certo.")
        return True
    st.error("❌ Não é esta. Pista: " + dica)
    return False


@st.fragment
def es_justificar(modo_prof):
    st.subheader("Parte 3 · Justificar")
    nota_objetivo(
        "Produto de uma matriz por uma matriz coluna, linha por coluna; as coordenadas de um segmento horizontal ou vertical; "
        "comprimento de um segmento; área de um triângulo, $\\tfrac12\\,b\\,h$.",
        "Passar de uma conjetura **testada em exemplos** para uma **justificação dedutiva**, válida para todos os pontos e todos os "
        "valores de $k$ e $l$, usando só definições e propriedades já conhecidas.",
        "O aluno completa os passos corretamente e escreve, com as suas palavras, uma justificação para cada conjetura da Parte 1.",
    )
    st.markdown(
        "Na Parte 1 viste **exemplos**. Agora vais mostrar que as tuas conjeturas valem **sempre**. "
        "Em cada uma, completa os passos e depois escreve a justificação completa."
    )

    st.markdown("#### (a) A imagem de um ponto")
    st.markdown("**Conjetura:** $M\\begin{bmatrix}x\\\\y\\end{bmatrix}=\\begin{bmatrix}kx\\\\ly\\end{bmatrix}$, com $M=\\begin{bmatrix}k&0\\\\0&l\\end{bmatrix}$.")
    _passo("es_j_a1", "A 1.ª coordenada da imagem é a 1.ª linha de $M$ vezes a coluna: $k\\cdot x + 0\\cdot y =$",
           ["$k+x$", "$kx$", "$kx+ly$", "$x$"], "$kx$",
           "calcula $k\\cdot x$ e $0\\cdot y$ e soma.")
    _passo("es_j_a2", "A 2.ª coordenada, com a 2.ª linha $(0,\\ l)$: $0\\cdot x + l\\cdot y =$",
           ["$l\\,y$", "$l+y$", "$ky$", "$y$"], "$l\\,y$",
           "a 2.ª linha de $M$ é $(0,\\ l)$.")
    st.text_area("Escreve a justificação (a)", key="es_j_ta", height=70)

    st.markdown("#### (b) A imagem de um segmento horizontal e de um vertical")
    st.markdown(
        "**Conjetura:** um segmento horizontal continua horizontal e o seu comprimento multiplica por $|k|$; "
        "um vertical continua vertical e o comprimento multiplica por $|l|$."
    )
    st.markdown("Considera o segmento horizontal de extremos $P(x_1,\\,c)$ e $Q(x_2,\\,c)$.")
    _passo("es_j_b1", "Por (a), a imagem de $P$ é $P'=(kx_1,\\ ?)$. A ordenada é",
           ["$c$", "$kc$", "$lc$", "$x_1$"], "$lc$",
           "usa (a): a ordenada fica multiplicada por $l$.")
    _passo("es_j_b2", "$P'$ e $Q'$ têm a mesma ordenada $lc$. Logo $P'Q'$ é um segmento",
           ["horizontal", "vertical", "oblíquo"], "horizontal",
           "pontos com a mesma ordenada estão numa reta de equação $y=\\text{constante}$.")
    _passo("es_j_b3", "O comprimento de $P'Q'$ é $|kx_2-kx_1| = $",
           ["$|k|\\,|x_2-x_1|$", "$|l|\\,|x_2-x_1|$", "$|k+l|\\,|x_2-x_1|$", "$|x_2-x_1|$"], "$|k|\\,|x_2-x_1|$",
           "põe $k$ em evidência.")
    st.markdown("Faz o mesmo para um segmento **vertical** de extremos $(c,\\ y_1)$ e $(c,\\ y_2)$.")
    st.text_area("Escreve a justificação (b), incluindo o caso vertical", key="es_j_tb", height=90)

    st.markdown("#### (c) A área")
    st.markdown(
        "**Conjetura:** a área da imagem é a área original multiplicada por $k\\,l$. "
        "Faz a justificação para o triângulo $ABC$, com base $AB$ (horizontal) e altura $CD$ (vertical)."
    )
    _passo("es_j_c1", "A área original é $\\tfrac12\\,|AB|\\,|CD|$. Na imagem, $A'B'$ é horizontal e $C'D'$ é vertical. "
           "$C'D'$ é a altura do triângulo $A'B'C'$ em relação à base $A'B'$?",
           ["sim, porque são perpendiculares", "não, porque se deformam", "só se $k=l$"], "sim, porque são perpendiculares",
           "um segmento horizontal e um vertical são sempre perpendiculares.")
    _passo("es_j_c2", "Por (b), quais são os comprimentos de $A'B'$ e de $C'D'$?",
           ["$k|AB|$ e $l|CD|$", "$l|AB|$ e $k|CD|$", "$k|AB|$ e $k|CD|$"], "$k|AB|$ e $l|CD|$",
           "a base é horizontal (fator $k$) e a altura é vertical (fator $l$).")
    _passo("es_j_c3", "A área da imagem é $\\tfrac12\\,|A'B'|\\,|C'D'| = \\tfrac12\\,(k|AB|)(l|CD|) =$",
           ["$(k+l)\\times$ área original", "$k\\,l\\times$ área original", "$\\dfrac{k}{l}\\times$ área original"],
           "$k\\,l\\times$ área original",
           "multiplica os fatores $k$ e $l$ e deixa $\\tfrac12|AB||CD|$.")
    st.text_area("Escreve a justificação (c)", key="es_j_tc", height=90)
    if modo_prof:
        st.info(
            "**Justificações modelo.**\n\n"
            "(a) $\\begin{bmatrix}k&0\\\\0&l\\end{bmatrix}\\begin{bmatrix}x\\\\y\\end{bmatrix}"
            "=\\begin{bmatrix}k\\,x+0\\,y\\\\0\\,x+l\\,y\\end{bmatrix}=\\begin{bmatrix}kx\\\\ly\\end{bmatrix}$.\n\n"
            "(b) Se $P(x_1,c)$ e $Q(x_2,c)$, então $P'(kx_1,\\,lc)$ e $Q'(kx_2,\\,lc)$ têm a mesma ordenada; $P'Q'$ é horizontal e "
            "$|P'Q'|=|kx_2-kx_1|=|k|\\,|x_2-x_1|$. Para $(c,y_1)$ e $(c,y_2)$: as imagens $(kc,\\,ly_1)$ e $(kc,\\,ly_2)$ têm a mesma abcissa, "
            "logo o segmento é vertical, de comprimento $|l|\\,|y_2-y_1|$.\n\n"
            "(c) A base $AB$ é horizontal e a altura $CD$ é vertical, logo $A'B'\\perp C'D'$ e $C'D'$ é a altura de $A'B'C'$. "
            "Então $\\text{área}'=\\tfrac12\\,|A'B'|\\,|C'D'|=\\tfrac12\\,(|k||AB|)(|l||CD|)=|k\\,l|\\cdot\\tfrac12|AB||CD|$. "
            "(Para figuras quaisquer, decompõe-se em triângulos desta forma, ou usa-se o quadrado unitário, que passa a um retângulo $k\\times l$.)"
        )


def modulo_escalonamento_draft(modo_prof):
    st.header("Escalonamento (rascunho)")
    st.markdown(
        "Tarefa curta sobre variações de tamanho. Em cada parte, a caixa 🎯 diz que conhecimento ou capacidade se pretende desenvolver, "
        "e que evidência a mostra. Estas caixas são para discutir; antes de dar a tarefa aos alunos, retiram-se ou passam ao modo professor."
    )
    st.markdown("**Mapa das partes**")
    st.markdown(tabela_md(pd.DataFrame({
        "Parte": ["1 · Explorar e conjeturar", "2 · Descobrir M", "3 · Justificar"],
        "Capacidade principal": [
            "Conjeturar sobre o efeito em pontos, segmentos e áreas",
            "Usar a conjetura sobre pontos para determinar as entradas de $M$",
            "Justificar dedutivamente as conjeturas da Parte 1",
        ],
        "Tipo de pensamento": ["Observação e generalização", "Dedução", "Prova"],
    })))
    es_explorar(modo_prof)
    st.divider()
    es_inverso(modo_prof)
    st.divider()
    es_justificar(modo_prof)


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📐 Transformações")
    st.markdown(
        "A matriz $X$ tem nas colunas os vértices do polígono: a coluna $i$, $\\mathbf{x}_i = \\begin{bmatrix} x_i \\\\ y_i \\end{bmatrix}$, "
        "é transformada em\n\n"
        "$$\\mathbf{x}_i' = M\\,\\mathbf{x}_i$$\n\n"
        "com $M$ uma matriz 2×2."
    )
    st.caption("O polígono de partida tem os vértices A(1,1), B(4,1) e C(2,3), guardados nas colunas da matriz $X$.")
    st.divider()
    modo_prof = st.checkbox("Modo professor (mostrar soluções)", value=False)

st.title("Como se move o triângulo?")
st.caption("Tarefa exploratória · pares · módulos autónomos de cerca de 30 minutos cada")

abas = st.tabs([
    "Laboratório",
    "A · Variações de tamanho",
    "B · Cisalhamentos",
    "C · Rotações",
    "D · Exercício",
    "escalonamento - draft",
])

# --------------------------------------------------------------------------
# Laboratório (setup da experiência)
# --------------------------------------------------------------------------
with abas[0]:
    st.header("Montar o laboratório")
    st.markdown(
        "Esta app é o teu laboratório. Escolhe um dos módulos — **A · Variações de tamanho**, **B · Cisalhamentos** ou "
        "**C · Rotações** — por qualquer ordem: cada um é autónomo e tem a sua exploração, desafios, conjeturas, justificação e caderno de bordo. "
        "Em cada módulo, **explora, conjetura, testa e justifica**, e regista tudo no caderno de bordo desse módulo, "
        "incluindo o que **não** funcionou."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("O polígono de partida, com **cada coluna a ser um vértice**:")
        st.latex(
            r"X = \begin{bmatrix} | & | & | \\ \mathbf{x}_1 & \mathbf{x}_2 & \mathbf{x}_3 \\ | & | & | \end{bmatrix}"
            r" = \begin{bmatrix} 1 & 4 & 2 \\ 1 & 1 & 3 \end{bmatrix}"
        )
        st.markdown(
            "- A coluna $\\mathbf{x}_1 = (1,1)$ é o ponto $A$, a coluna $\\mathbf{x}_2 = (4,1)$ é o ponto $B$ "
            "e a coluna $\\mathbf{x}_3 = (2,3)$ é o ponto $C$.\n"
            "- Dá-se o nome de **imagem** à figura que resulta de aplicar a transformação.\n"
            "- A aplicação mede também **área**, **lados** e **ângulos**, para testares as tuas conjeturas."
        )
    with c2:
        mostrar(desenhar([X], ["original"]))

with abas[1]:
    modulo_scaling(modo_prof)

with abas[2]:
    modulo_shearing(modo_prof)

with abas[3]:
    modulo_rotacoes(modo_prof)

with abas[4]:
    modulo_exercicio(modo_prof)

with abas[5]:
    modulo_escalonamento_draft(modo_prof)
