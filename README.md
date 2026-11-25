# Como se move o triângulo? (Streamlit)

## Executar localmente
```
pip install -r requirements.txt
streamlit run app.py
```

## Publicar (Streamlit Community Cloud)
1. Colocar `app.py` e `requirements.txt` num repositório do GitHub.
2. Em https://share.streamlit.io, criar uma app a partir do repositório (ficheiro principal: `app.py`).

O **Modo professor** (canto superior esquerdo da página, palavra-passe `ntem`) mostra as soluções e as notas para o professor.

## Estrutura
- **A · variação de tamanho**: tarefa curta em três partes:
  1. *Explora e responde* (resposta múltipla): o que a transformação faz a pontos, segmentos e áreas.
  2. *Descobrir M*: determinar as entradas de uma matriz para levar o triângulo ao alvo.
  3. *Justificar*: textos com lacunas (menus) que justificam as regras da Parte 1.

  No fim de cada separador, as respostas podem ser descarregadas num ficheiro Markdown.
- `.streamlit/config.toml` (opcional): tema de cores da app.
