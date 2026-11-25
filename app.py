"""
Como se move o triângulo?
Tarefa exploratória sobre transformações no plano com matrizes (Streamlit).

Separadores: A · variação de tamanho e B · cisalhamento (explorar e responder, descobrir M, justificar); C · rotações (vazio).

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


def mostrar(fig):
    """Mostra a figura como PNG a 120 dpi (mais leve e rápido do que o st.pyplot, que grava a 200 dpi)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    st.image(buf.getvalue())


def area(P):
    """Área do polígono (fórmula dos cordões)."""
    x, y = P[0], P[1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _f(v):
    """Número formatado para LaTeX (4 casas, sem «-0»)."""
    return f"{round(float(v), 4) + 0.0:g}"


def latex_mat(A):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    linhas = r" \\ ".join(" & ".join(_f(v) for v in linha) for linha in A)
    return r"\begin{bmatrix} " + linhas + r" \end{bmatrix}"


latex_matriz = latex_mat


def tabela_md(df):
    """Converte um DataFrame numa tabela Markdown (sem dependências extra)."""
    cols = list(df.columns)
    linhas = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        linhas.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in cols) + " |")
    return "\n".join(linhas)


# --------------------------------------------------------------------------
# A · Variação de tamanho — tarefa curta, com o objetivo de cada parte indicado (modo professor)
# --------------------------------------------------------------------------
def nota_objetivo(modo_prof, conhecimento, capacidade, evidencia):
    """Caixa para o professor: o que se pretende desenvolver na parte em causa (só no modo professor)."""
    if not modo_prof:
        return
    st.info(
        "🎯 **Professor · o que se pretende desenvolver nesta parte**\n\n"
        f"- **Conhecimento:** {conhecimento}\n"
        f"- **Capacidade:** {capacidade}\n"
        f"- **Evidência:** {evidencia}"
    )


def desenhar_variacao_tamanho(k, l, pontos=True, seg_h=False, seg_v=False, quadrado=False):
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
    st.subheader("Parte 1 · Explora e responde")
    nota_objetivo(
        modo_prof,
        "Uma variação de tamanho, $\\mathrm{diag}(k,l)$, leva $(x,y)$ a $(kx,\\,ly)$; um segmento horizontal continua horizontal, com o comprimento multiplicado por $k$, "
        "e um segmento vertical continua vertical, com o comprimento multiplicado por $l$; a área fica multiplicada por $k\\,l$.",
        "Identificar, por observação (gráfico e valores numéricos) e testando com vários valores de $k$ e $l$, o que a transformação faz a "
        "(a) cada ponto, (b) cada segmento horizontal e vertical e (c) à área. As opções erradas refletem ideias comuns (somar em vez de multiplicar, trocar $k$ e $l$).",
        "O aluno escolhe as três opções corretas depois de testar com diferentes valores de $k$ e $l$.",
    )
    st.markdown(
        "Nesta parte aplicas uma **matriz de variação de tamanho** $M=\\begin{bmatrix}k&0\\\\0&l\\end{bmatrix}$ a **cada ponto** do triângulo $ABC$: "
        "cada ponto $\\mathbf{x}=(x,y)$ é transformado em $\\mathbf{x}'=M\\,\\mathbf{x}$, e os pontos obtidos formam o triângulo imagem $A'B'C'$. "
        "Para desenhar a imagem basta transformar os três vértices, que são as colunas da matriz $X$:"
    )
    st.latex(r"X = \begin{bmatrix} | & | & | \\ \mathbf{x}_1 & \mathbf{x}_2 & \mathbf{x}_3 \\ | & | & | \end{bmatrix}"
             r" = \begin{bmatrix} 1 & 4 & 2 \\ 1 & 1 & 3 \end{bmatrix}"
             r" \ \longmapsto\ X' = M\,X = \begin{bmatrix} | & | & | \\ M\mathbf{x}_1 & M\mathbf{x}_2 & M\mathbf{x}_3 \\ | & | & | \end{bmatrix}")
    st.markdown(
        "Move $k$ e $l$, experimentando valores **iguais** e **diferentes**, maiores e menores do que 1. "
        "Escolhe em baixo **o que estás a investigar**: o gráfico e os dados mudam em conformidade. "
        "Responde às três perguntas."
    )
    foco = st.radio("A investigar", ["(a) Pontos", "(b) Segmentos", "(c) Área"], key="es_foco", horizontal=True)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        k = st.slider("k", 0.25, 3.0, 1.0, 0.25, key="es_k1")
        l = st.slider("l", 0.25, 3.0, 1.0, 0.25, key="es_l1")
        M = np.array([[k, 0.0], [0.0, l]])
        st.latex(rf"M = {latex_mat(M)}")
        X1 = M @ X
        if foco.startswith("(a)"):
            linhas = {"Ponto": [], "Original": [], "Imagem": []}
            for j in range(3):
                linhas["Ponto"].append(f"{chr(65 + j)} → {chr(65 + j)}′")
                linhas["Original"].append(f"({_f(X[0, j])}, {_f(X[1, j])})")
                linhas["Imagem"].append(f"({_f(X1[0, j])}, {_f(X1[1, j])})")
            st.markdown(tabela_md(pd.DataFrame(linhas)))
        elif foco.startswith("(b)"):
            D = np.array([X[0, 2], X[1, 0]])
            st.metric("Comprimento da base AB (original → imagem)",
                      f"{np.linalg.norm(X[:, 1] - X[:, 0]):.2f} → {np.linalg.norm(X1[:, 1] - X1[:, 0]):.2f}")
            st.metric("Comprimento da altura CD (original → imagem)",
                      f"{np.linalg.norm(X[:, 2] - D):.2f} → {np.linalg.norm(X1[:, 2] - M @ D):.2f}")
        else:
            st.metric("Área do triângulo (original → imagem)", f"{area(X):.2f} → {area(X1):.2f}")
    with c2:
        mostrar(desenhar_variacao_tamanho(k, l, pontos=foco.startswith("(a)"),
                                       seg_h=foco.startswith("(b)"), seg_v=foco.startswith("(b)"),
                                       quadrado=foco.startswith("(c)")))
    dica = "testa esta opção no gráfico, com outros valores de $k$ e $l$."
    st.markdown("**(a) Pontos.** Compara as coordenadas de cada ponto com as da sua imagem. Que se faz a $(x,y)$ para obter a imagem?")
    _passo("es_m1a", "Escolhe a opção correta:",
           ["$(x+k,\\ y+l)$", "$(kx,\\ ly)$", "$(kx,\\ y)$", "$(lx,\\ ky)$"], "$(kx,\\ ly)$", dica, horizontal=False)
    st.markdown("**(b) Segmentos.** Que acontece a um segmento horizontal (a base $AB$) e a um segmento vertical (a altura $CD$)?")
    _passo("es_m1b", "Escolhe a opção correta:",
           ["Continuam horizontal e vertical; o horizontal fica multiplicado por $k$ e o vertical por $l$.",
            "Continuam horizontal e vertical; o horizontal fica multiplicado por $l$ e o vertical por $k$.",
            "Continuam horizontal e vertical, mas os comprimentos não mudam.",
            "Ficam oblíquos."],
           "Continuam horizontal e vertical; o horizontal fica multiplicado por $k$ e o vertical por $l$.", dica, horizontal=False)
    st.markdown("**(c) Área.** Como se relaciona a área da imagem com a da figura original? *(O quadrado unitário pode ajudar.)*")
    _passo("es_m1c", "Escolhe a opção correta:",
           ["Fica multiplicada por $k+l$.", "Fica multiplicada por $k$.", "Fica multiplicada por $k\\,l$.", "Mantém-se."],
           "Fica multiplicada por $k\\,l$.", dica, horizontal=False)
    if modo_prof:
        st.info(
            "**Respostas esperadas:** (a) $(x,y)\\mapsto(k\\,x,\\ l\\,y)$: a abcissa multiplica por $k$ e a ordenada por $l$. "
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
        modo_prof,
        "$M$ leva $(x,y)$ a $(kx,\\,ly)$, isto é, $M=\\begin{bmatrix}k&0\\\\0&l\\end{bmatrix}$; logo $k=x'/x$ e $l=y'/y$ em qualquer vértice.",
        "Usar a conjetura da Parte 1 como **ferramenta**: a partir das coordenadas de $A$ e do ponto correspondente do alvo, "
        "determinar as entradas de $M$ (incluindo as que são nulas), e verificar com os outros vértices.",
        "O aluno preenche $M$ sem tentativa e erro prolongada e justifica-a com um vértice.",
    )
    alvos = [
        ("Alvo 1", np.diag([2.0, 1.0]) @ X, "$M=\\begin{bmatrix}2&0\\\\0&1\\end{bmatrix}$"),
        ("Alvo 2", np.diag([2.0, 0.5]) @ X, "$M=\\begin{bmatrix}2&0\\\\0&0{,}5\\end{bmatrix}$"),
    ]
    esc = st.selectbox("Desafio", [a[0] for a in alvos], key="es_d3")
    idx = [a[0] for a in alvos].index(esc)
    alvo = alvos[idx][1]
    st.markdown(
        "O triângulo alvo (a tracejado) é a imagem de $ABC$ por uma matriz $M$ que tens de descobrir. "
        "O gráfico mostra as coordenadas de $A$, $B$, $C$ e dos pontos correspondentes do alvo, $A^*$, $B^*$, $C^*$. "
        "**Usa a regra que identificaste na Parte 1** para encontrar as quatro entradas de $M$. "
        "Quando as preencheres, a imagem aparece a vermelho."
    )
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("**Entradas de $M$** (dispostas como na matriz)")
        ca, cb = st.columns(2)
        m11 = ca.number_input("linha 1, coluna 1", value=1.0, step=0.25, key=f"es_m3_{idx}_11", label_visibility="collapsed")
        m12 = cb.number_input("linha 1, coluna 2", value=0.0, step=0.25, key=f"es_m3_{idx}_12", label_visibility="collapsed")
        m21 = ca.number_input("linha 2, coluna 1", value=0.0, step=0.25, key=f"es_m3_{idx}_21", label_visibility="collapsed")
        m22 = cb.number_input("linha 2, coluna 2", value=1.0, step=0.25, key=f"es_m3_{idx}_22", label_visibility="collapsed")
        M = np.array([[m11, m12], [m21, m22]])
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


def _passo(chave, pergunta, opcoes, certa, dica, horizontal=True):
    """Passo de uma justificação guiada: escolha com resposta imediata. Devolve True se estiver certo."""
    st.markdown(pergunta)
    esc = st.radio("Escolhe", opcoes, index=None, key=chave, horizontal=horizontal, label_visibility="collapsed")
    if esc is None:
        return False
    if esc == certa:
        st.success("✅ Certo.")
        return True
    st.error("❌ Não é esta. Pista: " + dica)
    return False


SECOES_J = [
    {
        "titulo": "(a) A imagem de um ponto",
        "enunciado": "Queremos provar que $M\\begin{bmatrix}x\\\\y\\end{bmatrix}=\\begin{bmatrix}kx\\\\ly\\end{bmatrix}$, com $M=\\begin{bmatrix}k&0\\\\0&l\\end{bmatrix}$.",
        "linhas": [
            ("es_l_a1",
             ["Calculando o produto linha por coluna, a imagem de $(x,y)$ tem 1.ª coordenada",
              (["k·x", "k + x", "x", "k·x + l·y"], "k·x"),
              "e 2.ª coordenada",
              (["l·y", "l + y", "y", "k·x + l·y"], "l·y")],
             "a 1.ª linha de $M$ é $(k,\\ 0)$ e a 2.ª é $(0,\\ l)$."),
        ],
    },
    {
        "titulo": "(b) A imagem de um segmento horizontal e de um vertical",
        "enunciado": "Seja $PQ$ o segmento horizontal de extremos $P(x_1,\\,c)$ e $Q(x_2,\\,c)$. Por (a), $P'=(kx_1,\\ lc)$ e $Q'=(kx_2,\\ lc)$.",
        "linhas": [
            ("es_l_b1",
             ["$P'$ e $Q'$ têm a mesma ordenada; logo $P'Q'$ é um segmento",
              (["horizontal", "vertical", "oblíquo"], "horizontal")],
             "pontos com a mesma ordenada estão numa reta $y=\\text{constante}$."),
            ("es_l_b2",
             ["O comprimento de $P'Q'$ é",
              (["|k|·|x₂ − x₁|", "|l|·|x₂ − x₁|", "|k + l|·|x₂ − x₁|", "|x₂ − x₁|"], "|k|·|x₂ − x₁|")],
             "o comprimento de um segmento horizontal é a diferença, em módulo, das abcissas dos extremos: $|kx_2-kx_1|$."),
            ("es_l_b3",
             ["Analogamente, um segmento vertical continua",
              (["horizontal", "vertical", "oblíquo"], "vertical"),
              "e o seu comprimento fica multiplicado por",
              (["|k|", "|l|", "|k| + |l|", "1"], "|l|")],
             "agora é a ordenada que varia, e a ordenada fica multiplicada por $l$."),
        ],
    },
    {
        "titulo": "(c) A área",
        "enunciado": "Seja $ABC$ o triângulo de base $AB$ (horizontal) e altura $CD$ (vertical); a sua área é $\\tfrac12\\,|AB|\\,|CD|$.",
        "linhas": [
            ("es_l_c1",
             ["Na imagem, $A'B'$ é horizontal e $C'D'$ é vertical; são sempre",
              (["perpendiculares", "paralelos", "oblíquos"], "perpendiculares"),
              "Logo, $C'D'$ é a altura de $A'B'C'$."],
             "um segmento horizontal e um vertical formam um ângulo reto."),
            ("es_l_c2",
             ["Por (b), $|A'B'|=$",
              (["k·|AB|", "l·|AB|", "k·l·|AB|"], "k·|AB|"),
              "e $|C'D'|=$",
              (["k·|CD|", "l·|CD|", "k·l·|CD|"], "l·|CD|")],
             "a base é horizontal (fator $k$) e a altura é vertical (fator $l$)."),
            ("es_l_c3",
             ["Então a área da imagem, $\\tfrac12\\,|A'B'|\\,|C'D'|$, é igual à área original multiplicada por",
              (["k + l", "k·l", "k/l"], "k·l")],
             "multiplica os fatores e deixa $\\tfrac12\\,|AB|\\,|CD|$."),
        ],
    },
]


def _linha_lacunas(chave, partes, dica):
    """Frase com lacunas, uma linha por menu: o texto à esquerda, o menu (estreito) à direita e ✅/❌ no fim.
    O texto que vem depois da última lacuna aparece numa linha própria. Devolve True se todas as lacunas estiverem certas."""
    linhas, texto = [], ""
    for p in partes:
        if isinstance(p, str):
            texto = (texto + " " + p).strip()
        else:
            linhas.append((texto, p))
            texto = ""
    todas_certas, alguma_errada = True, False
    for i, (txt, (opcoes, certa)) in enumerate(linhas):
        c_txt, c_menu, c_marca = st.columns([3, 1.1, 0.25], vertical_alignment="center")
        c_txt.markdown(txt)
        esc = c_menu.selectbox(f"lacuna {i}", opcoes, index=None, placeholder="…", key=f"{chave}_{i}",
                               label_visibility="collapsed")
        if esc is None:
            todas_certas = False
        elif esc == certa:
            c_marca.markdown("✅")
        else:
            c_marca.markdown("❌")
            alguma_errada, todas_certas = True, False
    if texto:
        st.markdown(texto)
    if alguma_errada:
        st.caption("Pista: " + dica)
    return todas_certas


@st.fragment
def es_justificar(modo_prof):
    st.subheader("Parte 3 · Justificar")
    nota_objetivo(
        modo_prof,
        "Produto de uma matriz por uma matriz coluna, linha por linha; as coordenadas de um segmento horizontal ou vertical; "
        "comprimento de um segmento; área de um triângulo, $\\tfrac12\\,b\\,h$.",
        "Passar de uma regra **testada em exemplos** para uma **justificação dedutiva**, válida para todos os pontos e todos os "
        "valores de $k$ e $l$, completando cada passo com a propriedade que o justifica.",
        "O aluno completa corretamente as lacunas dos três textos.",
    )
    st.markdown(
        "Na Parte 1 descobriste algumas regras. Agora o objetivo é justificá-las. "
        "Completa cada texto escolhendo, em cada lacuna, a opção certa."
    )
    for sec in SECOES_J:
        with st.container(border=True):
            st.markdown(f"**{sec['titulo']}**")
            st.markdown(sec["enunciado"])
            for chave, partes, dica in sec["linhas"]:
                _linha_lacunas(chave, partes, dica)
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


RESPOSTAS_ES = [
    ("Parte 1 · (a) pontos", "es_m1a"),
    ("Parte 1 · (b) segmentos", "es_m1b"),
    ("Parte 1 · (c) área", "es_m1c"),
    ("Parte 2 · alvo 1, explicação", "es_t3_0"),
    ("Parte 2 · alvo 2, explicação", "es_t3_1"),
]


@st.fragment
def _exportar(prefixo, titulo, respostas, secoes, ficheiro):
    """Botões para guardar as respostas (opções escolhidas e textos) num ficheiro Markdown."""
    st.divider()
    st.subheader("Guardar as respostas")
    st.markdown("As respostas perdem-se se a página for atualizada. Escreve o nome do grupo, prepara o ficheiro e descarrega-o.")
    nome = st.text_input("Nome do grupo", key=f"{prefixo}_nome")
    if st.button("Preparar ficheiro com as minhas respostas", key=f"{prefixo}_prep"):
        linhas = [f"# {titulo} — {nome or 'sem nome'}", ""]
        for tit, chave in respostas:
            resp = st.session_state.get(chave, "")
            linhas += [f"## {tit}", resp if resp else "(sem resposta)", ""]
        for sec in secoes:
            linhas.append(f"## Parte 3 · {sec['titulo']}")
            for chave, partes, _ in sec["linhas"]:
                escolhidas = [st.session_state.get(f"{chave}_{i}") or "(vazio)" for i, p in enumerate(partes) if not isinstance(p, str)]
                linhas.append("- " + "; ".join(escolhidas))
            linhas.append("")
        st.download_button("Descarregar (.md)", "\n".join(linhas), file_name=ficheiro, key=f"{prefixo}_dl")


def modulo_variacao_tamanho(modo_prof):
    st.header("Variação de tamanho")
    st.markdown(
        "**Objetivo da tarefa:** caracterizar geometricamente a transformação por **variação de tamanho**, "
        "isto é, descobrir o que ela faz aos pontos, aos segmentos e à área de uma figura, e justificar porquê. "
        "Começas por explorar e responder, depois usas o que descobriste para determinar uma matriz, e acabas a justificar as regras."
    )
    if modo_prof:
        st.markdown("**Mapa das partes** *(modo professor)*")
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
    _exportar("es", "Variação de tamanho", RESPOSTAS_ES, SECOES_J, "variacao_tamanho_respostas.md")


# --------------------------------------------------------------------------
# B · Cisalhamento — tarefa curta, com a mesma estrutura da A (horizontal: M = [[1, q], [0, 1]])
# --------------------------------------------------------------------------
def desenhar_cisalhamento(q, pontos=True, seg_h=False, seg_v=False, quadrado=False):
    """Triângulo X e a sua imagem por [[1, q], [0, 1]], com ajudas visuais opcionais:
    deslocamento de cada ponto, base AB e altura CD com as suas imagens, quadrado unitário e a sua imagem."""
    M = np.array([[1.0, q], [0.0, 1.0]])
    X1 = M @ X
    D = np.array([X[0, 2], X[1, 0]])  # pé da altura traçada de C sobre a base AB
    D1 = M @ D
    Q = np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float)
    ext = [np.abs(X).max(), np.abs(X1).max(), 5.0]
    if quadrado:
        ext.append(np.abs(M @ Q).max())
    lim = float(np.ceil(max(ext))) + 1
    fig = Figure(figsize=(5.5, 5.5))
    ax = fig.subplots()
    if quadrado:
        for P, cor, nome in ((Q, "tab:blue", "quadrado unitário"), (M @ Q, "tab:red", "imagem do quadrado")):
            ax.fill(P[0], P[1], alpha=0.12, color=cor, zorder=0)
            ax.plot(np.append(P[0], P[0][0]), np.append(P[1], P[1][0]), "-", color=cor, lw=1.2, zorder=1, label=nome)
    if seg_h:
        for (p, r), cor, nome in (((X[:, 0], X[:, 1]), "tab:blue", "AB"), ((X1[:, 0], X1[:, 1]), "tab:red", "A′B′")):
            ax.plot([p[0], r[0]], [p[1], r[1]], "-", color=cor, lw=5, alpha=0.55, solid_capstyle="butt", zorder=3)
            ax.annotate(nome, (p + r) / 2, textcoords="offset points", xytext=(0, -16), ha="center",
                        color=cor, fontsize=10, fontweight="bold")
    if seg_v:
        for (p, r), cor, nome, d in (((D, X[:, 2]), "tab:blue", "CD", "D"), ((D1, X1[:, 2]), "tab:red", "C′D′", "D′")):
            ax.plot([p[0], r[0]], [p[1], r[1]], "-", color=cor, lw=5, alpha=0.55, solid_capstyle="butt", zorder=3)
            ax.plot(*p, "s", color=cor, ms=5, zorder=4)
            ax.annotate(d, p, textcoords="offset points", xytext=(6, -14), color=cor, fontsize=10)
            ax.annotate(nome, (p + r) / 2, textcoords="offset points", xytext=(8, 0), color=cor,
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
def ci_explorar(modo_prof):
    st.subheader("Parte 1 · Explora e responde")
    nota_objetivo(
        modo_prof,
        "Um cisalhamento horizontal, $\\begin{bmatrix}1&q\\\\0&1\\end{bmatrix}$, leva $(x,y)$ a $(x+qy,\\,y)$: os pontos de $Ox$ ficam fixos e os outros deslizam "
        "na horizontal, tanto mais quanto maior a ordenada. Um segmento horizontal mantém a direção e o comprimento; um vertical inclina-se e fica mais comprido. A área mantém-se ($\\det=1$).",
        "Identificar, por observação (gráfico e valores numéricos) e testando com vários valores de $q$, o que a transformação faz a "
        "(a) cada ponto, (b) cada segmento horizontal e vertical e (c) à área. As opções erradas refletem ideias comuns (somar $q$ às duas coordenadas, confundir com o cisalhamento vertical ou com a variação de tamanho).",
        "O aluno escolhe as três opções corretas depois de testar com diferentes valores de $q$.",
    )
    st.markdown(
        "Nesta parte aplicas uma **matriz de cisalhamento** $M=\\begin{bmatrix}1&q\\\\0&1\\end{bmatrix}$ a **cada ponto** do triângulo $ABC$: "
        "cada ponto $\\mathbf{x}=(x,y)$ é transformado em $\\mathbf{x}'=M\\,\\mathbf{x}$, e os pontos obtidos formam o triângulo imagem $A'B'C'$. "
        "Para desenhar a imagem basta transformar os três vértices, que são as colunas da matriz $X$:"
    )
    st.latex(r"X = \begin{bmatrix} | & | & | \\ \mathbf{x}_1 & \mathbf{x}_2 & \mathbf{x}_3 \\ | & | & | \end{bmatrix}"
             r" = \begin{bmatrix} 1 & 4 & 2 \\ 1 & 1 & 3 \end{bmatrix}"
             r" \ \longmapsto\ X' = M\,X = \begin{bmatrix} | & | & | \\ M\mathbf{x}_1 & M\mathbf{x}_2 & M\mathbf{x}_3 \\ | & | & | \end{bmatrix}")
    st.markdown(
        "Move $q$, experimentando valores **positivos**, **negativos** e **nulo**. "
        "Escolhe em baixo **o que estás a investigar**: o gráfico e os dados mudam em conformidade. "
        "Responde às três perguntas."
    )
    foco = st.radio("A investigar", ["(a) Pontos", "(b) Segmentos", "(c) Área"], key="ci_foco", horizontal=True)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        q = st.slider("q", -2.0, 2.0, 0.0, 0.25, key="ci_q1")
        M = np.array([[1.0, q], [0.0, 1.0]])
        st.latex(rf"M = {latex_mat(M)}")
        X1 = M @ X
        if foco.startswith("(a)"):
            linhas = {"Ponto": [], "Original": [], "Imagem": []}
            for j in range(3):
                linhas["Ponto"].append(f"{chr(65 + j)} → {chr(65 + j)}′")
                linhas["Original"].append(f"({_f(X[0, j])}, {_f(X[1, j])})")
                linhas["Imagem"].append(f"({_f(X1[0, j])}, {_f(X1[1, j])})")
            st.markdown(tabela_md(pd.DataFrame(linhas)))
        elif foco.startswith("(b)"):
            D = np.array([X[0, 2], X[1, 0]])
            st.metric("Comprimento da base AB (original → imagem)",
                      f"{np.linalg.norm(X[:, 1] - X[:, 0]):.2f} → {np.linalg.norm(X1[:, 1] - X1[:, 0]):.2f}")
            st.metric("Comprimento da altura CD (original → imagem)",
                      f"{np.linalg.norm(X[:, 2] - D):.2f} → {np.linalg.norm(X1[:, 2] - M @ D):.2f}")
        else:
            st.metric("Área do triângulo (original → imagem)", f"{area(X):.2f} → {area(X1):.2f}")
    with c2:
        mostrar(desenhar_cisalhamento(q, pontos=foco.startswith("(a)"),
                                      seg_h=foco.startswith("(b)"), seg_v=foco.startswith("(b)"),
                                      quadrado=foco.startswith("(c)")))
    dica = "testa esta opção no gráfico, com outros valores de $q$."
    st.markdown("**(a) Pontos.** Compara as coordenadas de cada ponto com as da sua imagem. Que se faz a $(x,y)$ para obter a imagem?")
    _passo("ci_m1a", "Escolhe a opção correta:",
           ["$(x+q,\\ y+q)$", "$(qx,\\ y)$", "$(x+qy,\\ y)$", "$(x,\\ y+qx)$"], "$(x+qy,\\ y)$", dica, horizontal=False)
    st.markdown("**(b) Segmentos.** Que acontece a um segmento horizontal (a base $AB$) e a um segmento vertical (a altura $CD$)?")
    _passo("ci_m1b", "Escolhe a opção correta:",
           ["Os dois continuam como estavam, apenas se deslocam.",
            "O horizontal fica oblíquo e o vertical mantém-se vertical.",
            "O horizontal mantém a direção e o comprimento; o vertical, se $q\\neq0$, deixa de ser vertical e fica mais comprido.",
            "Os dois ficam multiplicados por $q$."],
           "O horizontal mantém a direção e o comprimento; o vertical, se $q\\neq0$, deixa de ser vertical e fica mais comprido.",
           dica, horizontal=False)
    st.markdown("**(c) Área.** Como se relaciona a área da imagem com a da figura original? *(O quadrado unitário pode ajudar.)*")
    _passo("ci_m1c", "Escolhe a opção correta:",
           ["Fica multiplicada por $q$.", "Fica multiplicada por $1+q$.", "Mantém-se.", "Fica multiplicada por $q^2$."],
           "Mantém-se.", dica, horizontal=False)
    if modo_prof:
        st.info(
            "**Respostas esperadas:** (a) $(x,y)\\mapsto(x+q\\,y,\\ y)$: a ordenada mantém-se e a abcissa aumenta $q$ vezes a ordenada. "
            "(b) Um segmento horizontal mantém a direção e o comprimento (desloca-se $q\\,c$); um vertical inclina-se e fica mais comprido. "
            "(c) A área mantém-se (o quadrado unitário passa a um paralelogramo com a mesma base e a mesma altura)."
        )


@st.fragment
def ci_inverso(modo_prof):
    st.subheader("Parte 2 · Descobrir $M$")
    nota_objetivo(
        modo_prof,
        "$M$ leva $(x,y)$ a $(x+qy,\\,y)$, isto é, $M=\\begin{bmatrix}1&q\\\\0&1\\end{bmatrix}$; logo $q=(x'-x)/y$ em qualquer vértice com $y\\neq0$.",
        "Usar a regra da Parte 1 como **ferramenta**: a partir das coordenadas de $A$ e do ponto correspondente do alvo, "
        "determinar as entradas de $M$ (incluindo $1$ e $0$), e verificar com os outros vértices.",
        "O aluno preenche $M$ sem tentativa e erro prolongada e justifica-a com um vértice.",
    )
    alvos = [
        ("Alvo 1", np.array([[1.0, 1.0], [0.0, 1.0]]) @ X, "$M=\\begin{bmatrix}1&1\\\\0&1\\end{bmatrix}$"),
        ("Alvo 2", np.array([[1.0, -1.0], [0.0, 1.0]]) @ X, "$M=\\begin{bmatrix}1&-1\\\\0&1\\end{bmatrix}$"),
    ]
    esc = st.selectbox("Desafio", [a[0] for a in alvos], key="ci_d3")
    idx = [a[0] for a in alvos].index(esc)
    alvo = alvos[idx][1]
    st.markdown(
        "O triângulo alvo (a tracejado) é a imagem de $ABC$ por uma matriz $M$ que tens de descobrir. "
        "O gráfico mostra as coordenadas de $A$, $B$, $C$ e dos pontos correspondentes do alvo, $A^*$, $B^*$, $C^*$. "
        "**Usa a regra que identificaste na Parte 1** para encontrar as quatro entradas de $M$. "
        "Quando as preencheres, a imagem aparece a vermelho."
    )
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("**Entradas de $M$** (dispostas como na matriz)")
        ca, cb = st.columns(2)
        m11 = ca.number_input("linha 1, coluna 1", value=1.0, step=0.25, key=f"ci_m3_{idx}_11", label_visibility="collapsed")
        m12 = cb.number_input("linha 1, coluna 2", value=0.0, step=0.25, key=f"ci_m3_{idx}_12", label_visibility="collapsed")
        m21 = ca.number_input("linha 2, coluna 1", value=0.0, step=0.25, key=f"ci_m3_{idx}_21", label_visibility="collapsed")
        m22 = cb.number_input("linha 2, coluna 2", value=1.0, step=0.25, key=f"ci_m3_{idx}_22", label_visibility="collapsed")
        M = np.array([[m11, m12], [m21, m22]])
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
        "*(Que aconteceu à ordenada? E à abcissa?)* Porque é que a 2.ª linha de $M$ é $(0,\\ 1)$?"
    )
    st.text_area("A tua explicação", key=f"ci_t3_{idx}", height=80, label_visibility="collapsed")


SECOES_CI = [
    {
        "titulo": "(a) A imagem de um ponto",
        "enunciado": "Queremos provar que $M\\begin{bmatrix}x\\\\y\\end{bmatrix}=\\begin{bmatrix}x+qy\\\\y\\end{bmatrix}$, com $M=\\begin{bmatrix}1&q\\\\0&1\\end{bmatrix}$.",
        "linhas": [
            ("ci_l_a1",
             ["Calculando o produto linha por coluna, a imagem de $(x,y)$ tem 1.ª coordenada",
              (["x + q·y", "q·x + y", "x + y", "q·x"], "x + q·y"),
              "e 2.ª coordenada",
              (["y", "q·y", "x + y", "q·x + y"], "y")],
             "a 1.ª linha de $M$ é $(1,\\ q)$ e a 2.ª é $(0,\\ 1)$."),
        ],
    },
    {
        "titulo": "(b) A imagem de um segmento horizontal e de um vertical",
        "enunciado": "Seja $PQ$ o segmento horizontal de extremos $P(x_1,\\,c)$ e $Q(x_2,\\,c)$. Por (a), $P'=(x_1+qc,\\ c)$ e $Q'=(x_2+qc,\\ c)$.",
        "linhas": [
            ("ci_l_b1",
             ["$P'$ e $Q'$ têm a mesma ordenada; logo $P'Q'$ é um segmento",
              (["horizontal", "vertical", "oblíquo"], "horizontal")],
             "pontos com a mesma ordenada estão numa reta $y=\\text{constante}$."),
            ("ci_l_b2",
             ["O comprimento de $P'Q'$ é",
              (["|x₂ − x₁|", "|q|·|x₂ − x₁|", "|x₂ − x₁ + q·c|", "c·|x₂ − x₁|"], "|x₂ − x₁|")],
             "o comprimento é a diferença, em módulo, das abcissas: $|(x_2+qc)-(x_1+qc)|$."),
            ("ci_l_b3",
             ["Um segmento vertical de extremos $(c,\\ y_1)$ e $(c,\\ y_2)$ tem como imagens $(c+qy_1,\\ y_1)$ e $(c+qy_2,\\ y_2)$. "
              "Se $q\\neq0$, as abcissas das imagens são",
              (["iguais", "diferentes"], "diferentes"),
              "e por isso o segmento imagem é",
              (["horizontal", "vertical", "oblíquo"], "oblíquo")],
             "compara $c+qy_1$ com $c+qy_2$ quando $y_1\\neq y_2$."),
        ],
    },
    {
        "titulo": "(c) A área",
        "enunciado": "Seja $ABC$ o triângulo de base $AB$ (horizontal) e altura $CD$ (vertical); a sua área é $\\tfrac12\\,|AB|\\,|CD|$.",
        "linhas": [
            ("ci_l_c1",
             ["Na imagem, $A'B'$ continua horizontal e, por (b), o seu comprimento é",
              (["|AB|", "q·|AB|", "|AB|/q"], "|AB|")],
             "por (b), um segmento horizontal conserva o comprimento."),
            ("ci_l_c2",
             ["A altura de $A'B'C'$ em relação a $A'B'$ é a diferença das ordenadas de $C'$ e de $A'$, que a transformação não altera; logo mede",
              (["|CD|", "q·|CD|", "|CD|/q"], "|CD|")],
             "por (a), as ordenadas das imagens são iguais às originais."),
            ("ci_l_c3",
             ["Então a área da imagem, $\\tfrac12\\,|A'B'|\\,\\cdot$ altura, é igual à área original multiplicada por",
              (["1", "q", "1 + q"], "1")],
             "base e altura ficam iguais às originais."),
        ],
    },
]

RESPOSTAS_CI = [
    ("Parte 1 · (a) pontos", "ci_m1a"),
    ("Parte 1 · (b) segmentos", "ci_m1b"),
    ("Parte 1 · (c) área", "ci_m1c"),
    ("Parte 2 · alvo 1, explicação", "ci_t3_0"),
    ("Parte 2 · alvo 2, explicação", "ci_t3_1"),
]


@st.fragment
def ci_justificar(modo_prof):
    st.subheader("Parte 3 · Justificar")
    nota_objetivo(
        modo_prof,
        "Produto de uma matriz por uma matriz coluna, linha por linha; as coordenadas de um segmento horizontal ou vertical; "
        "comprimento de um segmento; área de um triângulo, $\\tfrac12\\,b\\,h$.",
        "Passar de uma regra **testada em exemplos** para uma **justificação dedutiva**, válida para todos os pontos e todos os "
        "valores de $q$, completando cada passo com a propriedade que o justifica.",
        "O aluno completa corretamente as lacunas dos três textos.",
    )
    st.markdown(
        "Na Parte 1 descobriste algumas regras. Agora o objetivo é justificá-las. "
        "Completa cada texto escolhendo, em cada lacuna, a opção certa."
    )
    for sec in SECOES_CI:
        with st.container(border=True):
            st.markdown(f"**{sec['titulo']}**")
            st.markdown(sec["enunciado"])
            for chave, partes, dica in sec["linhas"]:
                _linha_lacunas(chave, partes, dica)
    if modo_prof:
        st.info(
            "**Justificações modelo.**\n\n"
            "(a) $\\begin{bmatrix}1&q\\\\0&1\\end{bmatrix}\\begin{bmatrix}x\\\\y\\end{bmatrix}"
            "=\\begin{bmatrix}1\\,x+q\\,y\\\\0\\,x+1\\,y\\end{bmatrix}=\\begin{bmatrix}x+qy\\\\y\\end{bmatrix}$.\n\n"
            "(b) Se $P(x_1,c)$ e $Q(x_2,c)$, então $P'(x_1+qc,\\,c)$ e $Q'(x_2+qc,\\,c)$ têm a mesma ordenada; $P'Q'$ é horizontal e "
            "$|P'Q'|=|x_2-x_1|$. Para $(c,y_1)$ e $(c,y_2)$ com $y_1\\neq y_2$ e $q\\neq0$, as imagens $(c+qy_1,\\,y_1)$ e $(c+qy_2,\\,y_2)$ têm abcissas diferentes: "
            "o segmento é oblíquo (e mais comprido, de comprimento $\\sqrt{1+q^2}\\,|y_2-y_1|$).\n\n"
            "(c) $A'B'$ é horizontal, de comprimento $|AB|$; as ordenadas conservam-se, logo a altura de $A'B'C'$ é $|CD|$. "
            "Então $\\text{área}'=\\tfrac12\\,|AB|\\,|CD|$ = área original."
        )


def modulo_cisalhamento(modo_prof):
    st.header("Cisalhamento")
    st.markdown(
        "**Objetivo da tarefa:** caracterizar geometricamente a transformação por **cisalhamento** (horizontal), "
        "isto é, descobrir o que ela faz aos pontos, aos segmentos e à área de uma figura, e justificar porquê. "
        "Começas por explorar e responder, depois usas o que descobriste para determinar uma matriz, e acabas a justificar as regras."
    )
    if modo_prof:
        st.markdown("**Mapa das partes** *(modo professor)*")
        st.markdown(tabela_md(pd.DataFrame({
            "Parte": ["1 · Explora e responde", "2 · Descobrir M", "3 · Justificar"],
            "Capacidade principal": [
                "Identificar o efeito em pontos, segmentos e áreas",
                "Usar a regra sobre pontos para determinar as entradas de $M$",
                "Justificar dedutivamente as regras da Parte 1",
            ],
            "Tipo de pensamento": ["Observação e generalização", "Dedução", "Prova"],
        })))
    ci_explorar(modo_prof)
    st.divider()
    ci_inverso(modo_prof)
    st.divider()
    ci_justificar(modo_prof)
    _exportar("ci", "Cisalhamento", RESPOSTAS_CI, SECOES_CI, "cisalhamento_respostas.md")


# --------------------------------------------------------------------------
# Modo professor e título
# --------------------------------------------------------------------------
# Modo professor: canto superior esquerdo da página
col_prof, _ = st.columns([1, 3])
with col_prof:
    senha = st.text_input("Modo professor", type="password", key="senha_prof", placeholder="palavra-passe")
    modo_prof = senha == "ntem"
    if senha and not modo_prof:
        st.caption("Palavra-passe incorreta.")
    elif modo_prof:
        st.caption("Modo professor ativo: soluções e notas visíveis.")
        st.button("Desligar modo professor", key="desligar_prof",
                  on_click=lambda: st.session_state.update(senha_prof=""))

st.title("Como se move o triângulo?")
st.caption("Tarefa exploratória · pares")

abas = st.tabs(["A - variação de tamanho", "B - cisalhamento", "C - rotações"])

with abas[0]:
    modulo_variacao_tamanho(modo_prof)

with abas[1]:
    modulo_cisalhamento(modo_prof)

with abas[2]:
    st.header("Rotações")
