import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
from pathlib import Path
from io import BytesIO

st.set_page_config(
    page_title="Predição de Preço - Peças",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = Path('model.pkl')
    config_path = Path('model_config.json')

    if not model_path.exists():
        st.error("Model file not found. Please run 'python train_rf_model.py' first to generate the model.")
        st.stop()

    if not config_path.exists():
        st.error("Model configuration file not found. Please run 'python train_rf_model.py' first.")
        st.stop()

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        with open(config_path, 'r') as f:
            config = json.load(f)

        return model_data, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model_data, config = load_model()

st.markdown('<h1 class="main-header">Sistema de Predição de Preço - Peças Industriais</h1>', unsafe_allow_html=True)

st.sidebar.title("Informações do Modelo")
st.sidebar.markdown("---")

st.sidebar.metric("R²", f"{config['r_squared']:.4f}")
st.sidebar.metric("R² Ajustado", f"{config['r_squared_adj']:.4f}")
st.sidebar.metric("AIC", f"{config['aic']:.2f}")
st.sidebar.metric("Observações", config['n_observations'])

st.sidebar.markdown("---")
st.sidebar.markdown("### Variáveis")
st.sidebar.markdown(f"**Target:** {config['target']}")
st.sidebar.markdown("**Preditores:**")
for pred in config['predictors']:
    st.sidebar.markdown(f"- {pred}")

tab1, tab2, tab3 = st.tabs([
    "Predição Individual",
    "Predição em Lote",
    "Sobre o Modelo"
])

with tab1:
    st.header("Fazer Predição Individual")

    st.markdown("""
    Insira os valores das variáveis independentes para obter uma predição de preço.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Valores de Entrada")

        input_data = {}

        cols = st.columns(len(config['predictors']))

        for idx, var in enumerate(config['predictors']):
            with cols[idx]:
                if var == 'cavaco':
                    default_val = 20.0
                    min_val = 0.0
                    max_val = 100.0
                    step = 0.1
                elif var == 'peso_peca':
                    default_val = 2.0
                    min_val = 0.0
                    max_val = 50.0
                    step = 0.01
                elif var == 'comprimento':
                    default_val = 300.0
                    min_val = 0.0
                    max_val = 2000.0
                    step = 1.0
                else:
                    default_val = 1.0
                    min_val = 0.0
                    max_val = 1000.0
                    step = 0.1

                input_data[var] = st.number_input(
                    f"**{var}**",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=step,
                    help=f"Insira o valor de {var}"
                )

        if st.button("Fazer Predição", type="primary", use_container_width=True):
            try:
                new_data = pd.DataFrame([input_data])
                # Transforma e prediz usando modelo da faixa 0
                X = new_data[model_data['features']].values
                X_imp = model_data['imputer_faixa_0'].transform(X)
                prediction = model_data['model_faixa_0'].predict(X_imp)

                st.session_state['last_prediction'] = prediction[0]
                st.session_state['last_input'] = input_data

            except Exception as e:
                st.error(f"Erro na predição: {str(e)}")

    with col2:
        if 'last_prediction' in st.session_state:
            st.subheader("Resultado")

            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #1f77b4; margin: 0;">R$ {st.session_state['last_prediction']:.2f}</h2>
                <p style="margin: 0; color: #666;">Preço Predito</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Valores de Entrada")
            for var, value in st.session_state['last_input'].items():
                st.write(f"**{var}:** {value:.3f}")
        else:
            st.info("Insira os valores e clique em 'Fazer Predição'")

    st.markdown("---")
    st.subheader("Exemplos Rápidos")

    examples = [
        {"name": "Peça Pequena", "peso_mp": 1.0, "peso_peca": 0.5, "cavaco": 5.0, "comprimento": 150},
        {"name": "Peça Média", "peso_mp": 3.0, "peso_peca": 2.0, "cavaco": 20.0, "comprimento": 400},
        {"name": "Peça Grande", "peso_mp": 7.0, "peso_peca": 5.0, "cavaco": 50.0, "comprimento": 1000}
    ]

    cols = st.columns(len(examples))

    for idx, example in enumerate(examples):
        with cols[idx]:
            if st.button(f"{example['name']}", use_container_width=True):
                example_data = {k: v for k, v in example.items() if k != 'name'}
                try:
                    example_df = pd.DataFrame([example_data])
                    X = example_df[model_data['features']].values
                    X_imp = model_data['imputer_faixa_0'].transform(X)
                    pred = model_data['model_faixa_0'].predict(X_imp)
                    st.success(f"Preço estimado: R$ {pred[0]:.2f}")
                except Exception as e:
                    st.error(f"Erro: {str(e)}")

with tab2:
    st.header("Predição em Lote")

    st.markdown("""
    Faça upload de um arquivo Excel com múltiplas peças para obter predições em lote.

    **Formato esperado:** O arquivo deve conter as colunas: `{}`
    """.format(", ".join(config['predictors'])))

    uploaded_file = st.file_uploader(
        "Carregar arquivo Excel",
        type=['xlsx', 'xls'],
        help="Carregue um arquivo Excel com as variáveis necessárias"
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_excel(uploaded_file)

            st.success(f"Arquivo carregado: {batch_df.shape[0]} linhas, {batch_df.shape[1]} colunas")

            missing_vars = [var for var in config['predictors'] if var not in batch_df.columns]

            if missing_vars:
                st.error(f"Variáveis ausentes no arquivo: {', '.join(missing_vars)}")
                st.info(f"Colunas necessárias: {', '.join(config['predictors'])}")
                st.info(f"Colunas encontradas: {', '.join(batch_df.columns)}")
            else:
                st.subheader("Preview dos Dados")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("Gerar Predições", type="primary"):
                    with st.spinner("Processando predições..."):
                        try:
                            X = batch_df[config['predictors']].values
                            X_imp = model_data['imputer_faixa_0'].transform(X)
                            predictions = model_data['model_faixa_0'].predict(X_imp)
                            batch_df['preco_predito'] = predictions

                            st.success(f"{len(batch_df)} predições realizadas com sucesso!")

                            st.subheader("Resultados")
                            st.dataframe(batch_df, use_container_width=True)

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Preço Médio", f"R$ {batch_df['preco_predito'].mean():.2f}")
                            with col2:
                                st.metric("Preço Mínimo", f"R$ {batch_df['preco_predito'].min():.2f}")
                            with col3:
                                st.metric("Preço Máximo", f"R$ {batch_df['preco_predito'].max():.2f}")
                            with col4:
                                st.metric("Desvio Padrão", f"R$ {batch_df['preco_predito'].std():.2f}")

                            st.subheader("Distribuição dos Preços Preditos")
                            fig = px.histogram(
                                batch_df,
                                x='preco_predito',
                                nbins=30,
                                title="Distribuição dos Preços Preditos",
                                labels={'preco_predito': 'Preço Predito (R$)'},
                                color_discrete_sequence=['#1f77b4']
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                batch_df.to_excel(writer, index=False, sheet_name='Predições')

                            st.download_button(
                                label="Download Resultados (Excel)",
                                data=output.getvalue(),
                                file_name="predicoes_lote.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary"
                            )

                        except Exception as e:
                            st.error(f"Erro ao processar predições: {str(e)}")

        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")
    else:
        st.info("Carregue um arquivo Excel para começar")

        st.subheader("Exemplo de Formato")
        example_df = pd.DataFrame({
            'PN': ['PECA001', 'PECA002', 'PECA003'],
            'peso_mp': [4.5, 2.3, 0.8],
            'peso_peca': [2.918, 1.247, 0.526],
            'cavaco': [16.303, 65.515, 1.701],
            'comprimento': [370, 1740, 290]
        })
        st.dataframe(example_df, use_container_width=True)

with tab3:
    st.header("Sobre o Modelo")

    st.markdown("""
    Este sistema utiliza um modelo de **Regressão Linear Múltipla (OLS - Ordinary Least Squares)**
    pré-treinado para predizer o preço de peças industriais.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Métricas de Desempenho")

        metrics_df = pd.DataFrame({
            'Métrica': ['R²', 'R² Ajustado', 'AIC', 'BIC', 'Observações'],
            'Valor': [
                f"{config['r_squared']:.4f}",
                f"{config['r_squared_adj']:.4f}",
                f"{config['aic']:.2f}",
                f"{config['bic']:.2f}",
                str(config['n_observations'])
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
        **Interpretação:**
        - **R² = {config['r_squared']:.2%}**: O modelo explica {config['r_squared']:.1%} da variabilidade nos preços
        - **R² Ajustado**: Penaliza a adição de variáveis desnecessárias
        - **AIC/BIC**: Critérios de informação (menor = melhor)
        """)

    with col2:
        st.subheader("Configuração do Modelo")

        st.markdown(f"""
        **Fórmula:**
        ```
        {config['formula']}
        ```

        **Variável Dependente:** {config['target']}

        **Variáveis Independentes:**
        """)

        for pred in config['predictors']:
            st.markdown(f"- `{pred}`")

    st.markdown("---")
    st.subheader("Como Usar")

    st.markdown("""
    1. **Predição Individual**: Use a aba "Predição Individual" para estimar o preço de uma única peça
    2. **Predição em Lote**: Faça upload de um arquivo Excel com múltiplas peças na aba "Predição em Lote"
    3. **Interpretação**: O modelo fornece um preço estimado baseado nas características da peça

    """)

    st.markdown("---")
    st.info("""
    **Dica:** Para obter melhores predições, certifique-se de que os valores inseridos
    estão dentro do intervalo dos dados de treinamento.
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>Sistema de Predição v2.0</p>
        <p>Modelo pré-treinado</p>
    </div>
    """, unsafe_allow_html=True)
