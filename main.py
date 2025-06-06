import pandas as pd
import numpy as np
import pickle
import streamlit as st
from openai import OpenAI
from pycaret.classification import *  # load_model, predict_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import tempfile
import shap
import requests
import os
import seaborn as sns

# Função para interpretar o threshold via IA
def interpretar_threshold(comando_usuario):
    prompt = f"Extraia apenas o valor categórico do threshold ('baixo', 'medio', 'alto') baseado no comando: '{comando_usuario}'."
    try:
        token = st.secrets["gpt_token"]
        client = OpenAI(api_key=token)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Você é um assistente que extrai valores categóricos de thresholds ('baixo', 'medio', 'alto'). Quando o usuário solicitar um alteração de treshold, retorne um desses valores."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        resposta = response.choices[0].message.content.strip()
        if resposta == "baixo" or resposta == "medio" or resposta == "alto":
            print(resposta)
            return ["Novo threshold interpretado", resposta, True]
        else:
            print(resposta)

            return ["Por favor, informe um valor como 'baixo', 'medio', ou 'alto'", 'medio', False]
    except Exception as e:
        #st.error(f"Erro ao interpretar comando: {e}")
        return ["Por favor, informe um valor como 'baixo', 'medio', ou 'alto'", 'medio', False]


# Configurações iniciais do Streamlit
st.set_page_config(page_title='Risco de incêncio e queimadas',
                   page_icon='./images/logo_fiap.png',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title('Simulador - Risco de incêncio e queimadas')

# Integrantes
with st.expander('Integrantes (RM - NOME)', expanded=True):
    st.write('''
        555677 - Matheus Hungaro Fidelis  
        556389 - Pablo Menezes Barreto   
        556984 - Tiago Toshio Kumagai Gibo
    ''')

# Explicação
with st.expander('Descrição do App', expanded=False):
    st.write('''

        Este aplicativo tem como objetivo apoiar estratégias de marketing e vendas, permitindo simular, de forma simples e interativa, a chance de conversão de um cliente em potencial. A partir do preenchimento de informações de um cliente, o sistema utiliza um modelo de inteligência artificial treinado previamente para indicar se esse perfil tem maior ou menor propensão a adquirir um determinado produto.

        A plataforma também permite que o usuário ajuste o nível de rigor da análise (threshold) com um controle deslizante ou através de comandos em linguagem natural, o que torna a experiência mais personalizada e acessível.

        Além disso, o aplicativo oferece uma aba de análises comparativas, que destaca visualmente as principais diferenças entre os perfis de clientes que costumam comprar e os que não compram. Com gráficos claros e dinâmicos, é possível entender, por exemplo, quais características mais influenciam a decisão de compra — como renda, frequência de compras ou tempo desde a última interação.

        Essa solução foi pensada para ajudar equipes de vendas e marketing a tomar decisões mais informadas, segmentar melhor suas campanhas e aumentar as taxas de conversão, utilizando dados de forma estratégica e acessível.

    ''')

# Sidebar
st.sidebar.write('Configurações')
with st.sidebar:
    c1, c2 = st.columns([.3, .7])
    c1.image('./images/logo_fiap.png', width=100)
    c2.subheader('Case Ifood')
    database = 'CSV'

    if database == 'CSV':
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')

# Tela principal
if database == 'CSV':
    if file:
        # Carregamento do CSV
        Xtest = pd.read_csv(file)

        # Carregamento / instanciamento do modelo pkl
        mdl_rf = load_model('./pickle/pickle_rf_pycaret')

        # Predict do modelo
        ypred = predict_model(mdl_rf, data=Xtest, raw_score=True)

        with st.expander('Visualizar CSV carregado:', expanded=False):
            c1, _ = st.columns([2, 4])

            qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:',
                                   min_value=0,
                                   max_value=Xtest.shape[0],
                                   step=10,
                                   value=10)
            st.dataframe(Xtest.head(qtd_linhas))

        with st.expander('Visualizar Predições / Análises:', expanded=True):

            # Campo de texto para comando
            comando_usuario = st.text_input('Digite o comando para alterar o threshold (ex: "Alterar treshold para alto..."):',)

            # Threshold inicial
            treshold = 'medio'

            if comando_usuario:
                msg, treshold, success = interpretar_threshold(comando_usuario)

                if success:
                    st.success(f"{msg}: {treshold}")
                else:
                    st.warning(f"{msg}: {treshold}")

            # Slider para ajuste fino, já usando o threshold interpretado
            options = ['baixo', 'medio', 'alto']
            treshold = st.radio(
                label="Escolha uma opção:",
                options=options,
                horizontal=True,
                index=options.index(treshold),
            )

            predicoes, analises = st.tabs(["Predições", "Análises"])

            with predicoes:
                qtd_true = ypred.loc[ypred['prediction_label'] == treshold].shape[0]

                c1, _, c2, c3 = st.columns([.5, .1, .2, .2])
                c1.metric(f'Qtd risco {treshold}', value=qtd_true)


                # Função para colorir as predições
                def color_pred(val):
                    color = 'firebrick' if val == treshold else ''
                    return f'background-color: {color}'

                tipo_view = st.radio('Visualizar:', ('Completo', 'Apenas predições'))
                if tipo_view == 'Completo':
                    df_view = ypred.copy()
                else:
                    df_view = pd.DataFrame(ypred.iloc[:, -4].copy())

                st.dataframe(df_view.style.applymap(color_pred, subset=[f'prediction_label']))

                csv = df_view.to_csv(sep=',', decimal=',', index=True)
                st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
                st.download_button(label='Download CSV',
                                   data=csv,
                                   file_name='Predicoes.csv',
                                   mime='text/csv')

            with analises:
                st.markdown('###  Risco de incêndio e queimadas')
                fig, ax = plt.subplots(figsize=(4, 4), dpi=300, facecolor='#1F1D22')  # gráfico menor
                labels = [treshold, 'Outros']
                sizes = [qtd_true, len(ypred) - qtd_true]
                colors = ['darkgreen', 'firebrick']
                explode = (0, 0.05)

                ax.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    explode=explode,
                    shadow=True,
                    textprops={'fontsize': 6, 'color': 'white'},      # ← aqui controla o tamanho da fonte
                    labeldistance=1.1                # ← distância do rótulo até o centro
                )
                ax.set_facecolor('white')
                ax.axis('equal')
                col1, col2 = st.columns([0.5, 0.5])
                with col1:
                    st.pyplot(fig)


    else:
        st.warning('Arquivo CSV não foi carregado.')
