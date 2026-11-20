"""
Como se move o triângulo?
Tarefa exploratória sobre transformações no plano com matrizes (Streamlit).

Executar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import streamlit as st

st.set_page_config(page_title="Como se move o triângulo?", page_icon="📐", layout="wide")

# --------------------------------------------------------------------------
# Dados e funções auxiliares
# --------------------------------------------------------------------------
# Polígono de partida: cada COLUNA é um vértice (x; y) -> A(1,1), B(4,1), C(2,3)
T = np.array([[1, 4, 2],
              [1, 1, 3]], dtype=float)

CORES = ["tab:blue", "tab:red", "tab:green", "tab:orange"]


def desenhar(figuras, nomes, alvo=None, lim=8):
    """Devolve uma figura matplotlib com os polígonos (matrizes 2×n, colunas = vértices)."""
    fig = Figure(figsize=(5.5, 5.5))  # Figure direto (sem pyplot): seguro com vários utilizadores em simultâneo
    ax = fig.subplots()
    for i, (X, nome) in enumerate(zip(figuras, nomes)):
        xs, ys = np.append(X[0], X[0][0]), np.append(X[1], X[1][0])
        ax.fill(xs, ys, alpha=0.2, color=CORES[i % 4])
        ax.plot(xs, ys, "o-", color=CORES[i % 4], label=nome)
    if alvo is not None:
        xs, ys = np.append(alvo[0], alvo[0][0]), np.append(alvo[1], alvo[1][0])
        ax.plot(xs, ys, "k--", label="alvo")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    ax.legend(loc="upper left")
    return fig


def mostrar(fig):
    st.pyplot(fig)


def area(X):
    """Área do polígono (fórmula dos cordões)."""
    x, y = X[0], X[1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def lados(X):
    """Comprimentos dos lados, pela ordem dos vértices."""
    return np.linalg.norm(X - np.roll(X, -1, axis=1), axis=0)


def mostrar_medidas(X, X1):
    """Mostra a comparação entre a figura X e a sua imagem X1."""
    a0, a1 = area(X), area(X1)
    ang = np.degrees(np.arctan2(X1[1, 0], X1[0, 0]) - np.arctan2(X[1, 0], X[0, 0])) % 360
    c1, c2, c3 = st.columns(3)
    c1.metric("Área (original → imagem)", f"{a0:.2f} → {a1:.2f}", f"razão {a1 / a0:.3f}" if a0 else None, delta_color="off")
    c2.metric("Ângulo O→A até O→A'", f"{ang:.1f}°", help="Só faz sentido se a origem for o centro da transformação.")
    c3.write("**Lados**")
    c3.write(f"{np.round(lados(X), 3).tolist()}  \n→ {np.round(lados(X1), 3).tolist()}")


def latex_matriz(M):
    (p, q), (r, s) = M
    return rf"\begin{{bmatrix}} {p:g} & {q:g} \\ {r:g} & {s:g} \end{{bmatrix}}"


def latex_coluna(v):
    return rf"\begin{{bmatrix}} {v[0]:g} \\ {v[1]:g} \end{{bmatrix}}"


def entrada_matriz(prefixo, M0=((1.0, 0.0), (0.0, 1.0)), v0=(0.0, 0.0)):
    """Caixas numéricas para M (2×2) e v (2×1). Devolve (M, v)."""
    st.markdown("**Matriz M**")
    c1, c2 = st.columns(2)
    p = c1.number_input("p", value=float(M0[0][0]), step=0.1, key=f"{prefixo}_p")
    q = c2.number_input("q", value=float(M0[0][1]), step=0.1, key=f"{prefixo}_q")
    r = c1.number_input("r", value=float(M0[1][0]), step=0.1, key=f"{prefixo}_r")
    s = c2.number_input("s", value=float(M0[1][1]), step=0.1, key=f"{prefixo}_s")
    st.markdown("**Vetor v**")
    c3, c4 = st.columns(2)
    a = c3.number_input("a", value=float(v0[0]), step=0.5, key=f"{prefixo}_a")
    b = c4.number_input("b", value=float(v0[1]), step=0.5, key=f"{prefixo}_b")
    return np.array([[p, q], [r, s]]), np.array([[a], [b]])


def tabela_md(df):
    """Converte um DataFrame numa tabela Markdown (sem dependências extra)."""
    cols = list(df.columns)
    linhas = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        linhas.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in cols) + " |")
    return "\n".join(linhas)


# --------------------------------------------------------------------------
# Desafios (a solução só é mostrada no modo professor)
# --------------------------------------------------------------------------
DESAFIOS = [
    {
        "titulo": "1. Deslocar sem deformar",
        "texto": "Desloca a figura 3 unidades para a esquerda e 2 para cima, **sem a deformar**.",
        "alvo": np.array([[-2, 1, -1], [3, 3, 5]], dtype=float),
        "solucao": r"$M = I$ e $v = (-3,\,2)$.",
    },
    {
        "titulo": "2. Ampliar para o dobro",
        "texto": "Amplia a figura para o dobro, **mantendo a forma** (centro: origem).",
        "alvo": np.array([[2, 8, 4], [2, 2, 6]], dtype=float),
        "solucao": r"$M = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$.",
    },
    {
        "titulo": "3. Alargar na horizontal",
        "texto": "Alarga a figura na horizontal (fator 2) e deixa-a igual na vertical.",
        "alvo": np.array([[2, 8, 4], [1, 1, 3]], dtype=float),
        "solucao": r"$M = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$.",
    },
    {
        "titulo": "4. Refletir em Oy",
        "texto": "Reflete a figura em relação ao eixo $Oy$. *(Isto já é scaling? Discute.)*",
        "alvo": np.array([[-1, -4, -2], [1, 1, 3]], dtype=float),
        "solucao": r"$M = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$ (scaling com fator negativo).",
    },
    {
        "titulo": "5. Inclinar mantendo A e B",
        "texto": (
            "Inclina a figura \"para a direita\", mantendo os pontos $A$ e $B$ **exatamente onde estão**. "
            "*Dica: só com $M$ consegues? E se usares também $v$?*"
        ),
        "alvo": np.array([[1, 4, 4], [1, 1, 3]], dtype=float),
        "solucao": (
            r"$M = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ e $v = (-1,\,0)$. "
            "Só com $M$, $A$ e $B$ não ficam fixos (a base está em $y=1$)."
        ),
    },
    {
        "titulo": "6. Rotações",
        "texto": (
            "Roda a figura em torno da origem. O ângulo de rotação, em radianos, é $\\alpha = c\\,\\pi$ e o que escolhes é o "
            "valor de $c$ (positivo = sentido contrário ao dos ponteiros do relógio). "
            "Começa por fazer coincidir a imagem com o alvo *(que fração de $\\pi$ corresponde a um quarto de volta?)*. "
            "Depois experimenta outros valores de $c$, como $0{,}25$, $-1/6$ (aprox. $-0{,}167$) ou $1$: para esses não há alvo, "
            "usa as medições (ângulo, lados, área)."
        ),
        "alvo": np.array([[-1, -1, -3], [1, 4, 2]], dtype=float),
        "solucao": (
            r"$c = 0{,}5$, isto é, $\alpha = \pi/2$ rad, que dá $M = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$; em geral "
            r"$\begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix}$ "
            r"(as colunas são as imagens de $(1,0)$ e $(0,1)$)."
        ),
    },
    {
        "titulo": "7. Desafio aberto",
        "texto": "Encontra uma matriz $M$ que **não** seja nenhuma das anteriores e descreve o que faz à figura.",
        "alvo": None,
        "solucao": "Resposta livre.",
    },
]

# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📐 Transformações")
    st.markdown(
        "Cada ponto $X = \\begin{bmatrix} x \\\\ y \\end{bmatrix}$ é transformado em\n\n"
        "$$X' = M\\,X + v$$\n\n"
        "com $M$ uma matriz 2×2 e $v$ uma matriz coluna."
    )
    st.caption("O polígono de partida tem os vértices A(1,1), B(4,1) e C(2,3), guardados nas colunas de uma matriz.")
    st.divider()
    modo_prof = st.checkbox("Modo professor (mostrar soluções)", value=False)

st.title("Como se move o triângulo?")
st.caption("Tarefa exploratória · pares · cerca de 90 minutos")

abas = st.tabs([
    "0 · Laboratório",
    "1 · Exploração livre",
    "2 · Caça às transformações",
    "3 · Conjeturar e testar",
    "4 · Justificar",
    "5 · Síntese",
    "📓 Caderno de bordo",
])

# --------------------------------------------------------------------------
# Momento 0
# --------------------------------------------------------------------------
with abas[0]:
    st.header("Momento 0 — Montar o laboratório")
    st.markdown(
        "Esta app vai ser o teu laboratório. Segue a ordem dos separadores, **explora, conjetura, testa e justifica**. "
        "Regista tudo no **Caderno de bordo**, incluindo o que **não** funcionou."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("O polígono de partida, com **cada coluna a ser um vértice**:")
        st.latex(r"T = " + rf"\begin{{bmatrix}} 1 & 4 & 2 \\ 1 & 1 & 3 \end{{bmatrix}}")
        st.markdown(
            "- A 1.ª coluna é o ponto $A(1,1)$, a 2.ª é $B(4,1)$ e a 3.ª é $C(2,3)$.\n"
            "- Dá-se o nome de **imagem** à figura que resulta de aplicar a transformação.\n"
            "- A aplicação mede também **área**, **lados** e **ângulos**, para testares as tuas conjeturas."
        )
    with c2:
        mostrar(desenhar([T], ["original"]))

# --------------------------------------------------------------------------
# Momento 1
# --------------------------------------------------------------------------
with abas[1]:
    st.header("Momento 1 — Exploração livre (15 min)")
    st.markdown(
        "Esta é a tua \"máquina\" de transformar figuras:"
    )
    st.latex(r"X' = M\,X + v, \qquad M = \begin{bmatrix} p & q \\ r & s \end{bmatrix}, \quad v = \begin{bmatrix} a \\ b \end{bmatrix}")
    st.markdown(
        "**Primeiro**, com $M$ igual à identidade, mexe **só** em $a$ e $b$. "
        "**Depois**, com $a = b = 0$, altera **um seletor de $M$ de cada vez**. "
        "Regista o que observas no Caderno de bordo."
    )
    cs, cg = st.columns([1, 1.3])
    with cs:
        st.subheader("Vetor v")
        a = st.slider("a", -5.0, 5.0, 0.0, 0.5, key="m1_a")
        b = st.slider("b", -5.0, 5.0, 0.0, 0.5, key="m1_b")
        st.subheader("Matriz M")
        p = st.slider("p", -3.0, 3.0, 1.0, 0.1, key="m1_p")
        q = st.slider("q", -3.0, 3.0, 0.0, 0.1, key="m1_q")
        r = st.slider("r", -3.0, 3.0, 0.0, 0.1, key="m1_r")
        s = st.slider("s", -3.0, 3.0, 1.0, 0.1, key="m1_s")
        M1 = np.array([[p, q], [r, s]])
        v1 = np.array([[a], [b]])
        st.latex(rf"M = {latex_matriz(M1)}, \quad v = {latex_coluna((a, b))}")
    with cg:
        T1 = M1 @ T + v1
        mostrar(desenhar([T, T1], ["original", "imagem"]))

# --------------------------------------------------------------------------
# Momento 2
# --------------------------------------------------------------------------
with abas[2]:
    st.header("Momento 2 — Caça às transformações (30 min)")
    st.markdown(
        "Em cada desafio, **descobre** os valores de $M$ e de $v$ que produzem o efeito pedido. "
        "Se houver uma figura **alvo** (tracejada), tenta fazer coincidir a imagem com ela. "
        "Regista o que tentaste, inclusive o que **não** funcionou."
    )
    escolha = st.selectbox("Desafio", [d["titulo"] for d in DESAFIOS], key="m2_desafio")
    d = next(x for x in DESAFIOS if x["titulo"] == escolha)
    idx = DESAFIOS.index(d)
    st.markdown(d["texto"])

    ce, cg = st.columns([1, 1.3])
    with ce:
        if idx == 5:
            # Desafio 6: o input é o ângulo de rotação (radianos); a app calcula M
            c = st.number_input("c  (ângulo de rotação α = c·π radianos)", value=0.0, step=0.05, format="%.3f",
                                key="d5_c")
            alfa = c * np.pi
            st.caption(f"α = {c:g}·π ≈ {alfa:.4f} rad")
            M2 = np.array([[np.cos(alfa), -np.sin(alfa)],
                           [np.sin(alfa),  np.cos(alfa)]])
            v2 = np.zeros((2, 1))
            st.markdown("Cada elemento de $M$ é uma **função do ângulo** $\\alpha$:")
            st.latex(r"M(\alpha) = \begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix}")
            c_, s_ = round(float(np.cos(alfa)), 4) + 0.0, round(float(np.sin(alfa)), 4) + 0.0  # +0.0 evita «-0.0000»
            st.markdown("Para o valor de $\\alpha$ escolhido:")
            st.latex(
                rf"\cos\alpha = {c_:.4f}, \qquad \sin\alpha = {s_:.4f}"
            )
            st.latex(
                rf"M = \begin{{bmatrix}} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{{bmatrix}}"
                rf" = {latex_matriz(np.round(M2, 4) + 0.0)}"
            )
        else:
            M2, v2 = entrada_matriz(f"d{idx}")
    with cg:
        T2 = M2 @ T + v2
        mostrar(desenhar([T, T2], ["original", "imagem"], alvo=d["alvo"]))
        if d["alvo"] is not None:
            if np.allclose(T2, d["alvo"]):
                st.success("✅ Coincide com o alvo!")
            else:
                st.error("❌ Ainda não coincide com o alvo.")
    mostrar_medidas(T, T2)

    if modo_prof:
        st.info("**Solução:** " + d["solucao"])

    if idx == 5:
        st.divider()
        st.subheader("Pergunta de pesquisa (à escolha do grupo)")
        st.markdown(
            "A app usa $M(\\alpha)$ com $\\cos\\alpha$ e $\\sin\\alpha$ nas entradas. **Porque é que isto faz sentido?**\n\n"
            "- Observa para onde vão os pontos $(1,0)$ e $(0,1)$ e compara com as **colunas** de $M$. "
            "Como se relacionam essas imagens com o círculo trigonométrico?\n"
            "- Experimenta $\\alpha$ e $-\\alpha$ (por exemplo $c = 0{,}25$ e $c = -0{,}25$): o que acontece às entradas de $M$? "
            "Que rotação \"desfaz\" a rotação de ângulo $\\alpha$?\n"
            "- Experimenta $c = 0$, $c = 1$ e $c = 2$ (isto é, $\\alpha = 0$, $\\pi$ e $2\\pi$). Que matrizes obténs?"
        )
        e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        st.latex(
            rf"M\begin{{bmatrix}} 1 \\ 0 \end{{bmatrix}} = {latex_coluna(np.round(M2 @ e1, 4) + 0.0)}, \qquad "
            rf"M\begin{{bmatrix}} 0 \\ 1 \end{{bmatrix}} = {latex_coluna(np.round(M2 @ e2, 4) + 0.0)}"
        )

# --------------------------------------------------------------------------
# Momento 3
# --------------------------------------------------------------------------
with abas[3]:
    st.header("Momento 3 — Conjeturar e testar (20 min)")
    st.markdown(
        "Cada grupo formula **por escrito** pelo menos **três conjeturas**, no formato:\n\n"
        "> *\"Quando [condição sobre os números], a figura [efeito].\"*\n\n"
        "**Sugestões de pontos a investigar:**\n"
        "- Qual é o papel de $v$ numa translação? Existe um $v$ que \"desfaz\" outro?\n"
        "- Quando é que a **área** da imagem é igual à da figura original? E quando é o dobro?\n"
        "- Quando é que os **comprimentos** e os **ângulos** se mantêm?\n"
        "- Que matrizes $M$ deixam a origem fixa? Há outros pontos fixos?\n"
        "- A ordem importa? Compara $MX + v$ com $M(X + v)$ (abaixo).\n\n"
        "Para cada conjetura, escreve **como a testaste** e **se resistiu**. "
        "Procura um contraexemplo antes de a dares por boa."
    )

    st.subheader("Laboratório de conjeturas")
    ce, cg = st.columns([1, 1.3])
    with ce:
        M3, v3 = entrada_matriz("lab")
    with cg:
        T3 = M3 @ T + v3
        mostrar(desenhar([T, T3], ["original", "imagem"]))
    mostrar_medidas(T, T3)

    st.divider()
    st.subheader("A ordem importa?   MX + v   versus   M(X + v)")
    ce, cg = st.columns([1, 1.3])
    with ce:
        Mo, vo = entrada_matriz("ord", M0=((0.0, -1.0), (1.0, 0.0)), v0=(2.0, 1.0))
    with cg:
        T_a = Mo @ T + vo
        T_b = Mo @ (T + vo)
        mostrar(desenhar([T, T_a, T_b], ["original", "M·X + v", "M·(X + v)"]))
        st.write("**As duas imagens são iguais?**", "Sim ✅" if np.allclose(T_a, T_b) else "Não ❌")

# --------------------------------------------------------------------------
# Momento 4
# --------------------------------------------------------------------------
with abas[4]:
    st.header("Momento 4 — Justificar (10 min)")
    st.markdown(
        "Escolhe **uma** conjetura que tenha resistido e justifica-a com cálculo. Por exemplo, efetua à mão"
    )
    st.latex(
        r"\begin{bmatrix} p & q \\ r & s \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix}"
        r"\quad\text{e}\quad"
        r"\begin{bmatrix} p & q \\ r & s \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix}"
    )
    st.markdown("e explica o que representam as **colunas** de $M$. Podes confirmar o teu cálculo numericamente:")
    ce, cg = st.columns([1, 1.3])
    with ce:
        M4, _ = entrada_matriz("just")
    with cg:
        e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        st.latex(
            rf"{latex_matriz(M4)}\begin{{bmatrix}} 1 \\ 0 \end{{bmatrix}} = {latex_coluna(M4 @ e1)}"
        )
        st.latex(
            rf"{latex_matriz(M4)}\begin{{bmatrix}} 0 \\ 1 \end{{bmatrix}} = {latex_coluna(M4 @ e2)}"
        )
    if modo_prof:
        st.info(
            "**Conclusão esperada:** a 1.ª coluna de $M$ é a imagem de $(1,0)$ e a 2.ª é a imagem de $(0,1)$. "
            "Em geral, $M\\begin{bmatrix} x \\\\ y \\end{bmatrix} = \\begin{bmatrix} px + qy \\\\ rx + sy \\end{bmatrix}$."
        )

# --------------------------------------------------------------------------
# Momento 5
# --------------------------------------------------------------------------
with abas[5]:
    st.header("Momento 5 — Partilha e síntese (turma)")
    st.markdown(
        "Cada grupo apresenta uma descoberta e um contraexemplo ou dificuldade. "
        "A turma constrói o quadro-resumo coletivo **depois** de ouvir os grupos. "
        "O professor formaliza a terminologia (translação, scaling, shearing, rotação) e a notação."
    )
    st.markdown("O quadro-resumo preenche-se no separador **📓 Caderno de bordo**.")
    if modo_prof:
        st.info(
            "- A razão das áreas no scaling é $|k \\cdot l|$.\n"
            "- O cisalhamento conserva a área, mas não comprimentos nem ângulos.\n"
            "- A rotação conserva comprimentos, ângulos e área; a inversa de $R(\\alpha)$ é $R(-\\alpha)$.\n"
            "- Em geral, $MX + v \\neq M(X + v)$."
        )

# --------------------------------------------------------------------------
# Caderno de bordo
# --------------------------------------------------------------------------
with abas[6]:
    st.header("Caderno de bordo")
    st.caption("O caderno fica guardado enquanto a página não for recarregada. Descarrega-o no fim.")

    nomes_grupo = st.text_input("Nomes do grupo", key="cb_grupo")

    st.subheader("Momento 1 — O que observei")
    df1 = pd.DataFrame({
        "Seletor alterado": ["a ou b", "p", "q", "r", "s"],
        "O que observei": [""] * 5,
        "Que conjetura posso fazer?": [""] * 5,
    })
    ed1 = st.data_editor(df1, hide_index=True, use_container_width=True, key="cb_m1",
                         disabled=["Seletor alterado"])

    st.subheader("Momento 2 — Desafios")
    df2 = pd.DataFrame({
        "Desafio": [d["titulo"] for d in DESAFIOS],
        "M e v que usei": [""] * len(DESAFIOS),
        "Resultou? O que tentei (e não funcionou)": [""] * len(DESAFIOS),
    })
    ed2 = st.data_editor(df2, hide_index=True, use_container_width=True, key="cb_m2",
                         disabled=["Desafio"])

    st.subheader("Momento 3 — Conjeturas")
    conjeturas = []
    for i in range(1, 4):
        with st.expander(f"Conjetura {i}", expanded=(i == 1)):
            c = st.text_area("Quando [condição], a figura [efeito].", key=f"cb_c{i}")
            t = st.text_area("Como a testei (valores usados)", key=f"cb_t{i}")
            res = st.radio("Resistiu?", ["—", "Sim", "Não (contraexemplo)"], horizontal=True, key=f"cb_r{i}")
            conjeturas.append((c, t, res))

    st.subheader("Momento 4 — Justificação")
    just = st.text_area("Conjetura escolhida e justificação com cálculo", key="cb_just", height=140)

    st.subheader("Momento 5 — Quadro-resumo")
    df5 = pd.DataFrame({
        "Tipo de transformação": ["Translação", "Scaling", "Cisalhamento (shearing)", "Rotação"],
        "Como a reconhecemos": [""] * 4,
        "Matriz / operação": [""] * 4,
        "O que se conserva": [""] * 4,
    })
    ed5 = st.data_editor(df5, hide_index=True, use_container_width=True, key="cb_m5",
                         disabled=["Tipo de transformação"])

    # Exportação em Markdown
    md = ["# Caderno de bordo — Como se move o triângulo?", "", f"**Grupo:** {nomes_grupo}", "",
          "## Momento 1", "", tabela_md(ed1), "",
          "## Momento 2", "", tabela_md(ed2), "",
          "## Momento 3 — Conjeturas", ""]
    for i, (c, t, res) in enumerate(conjeturas, start=1):
        md += [f"**Conjetura {i}:** {c}", "", f"- Teste: {t}", f"- Resistiu? {res}", ""]
    md += ["## Momento 4 — Justificação", "", just, "", "## Momento 5 — Quadro-resumo", "", tabela_md(ed5), ""]
    st.download_button("⬇️ Descarregar caderno (Markdown)", "\n".join(md),
                       file_name="caderno_de_bordo.md", mime="text/markdown")