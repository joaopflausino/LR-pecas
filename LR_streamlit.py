import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from io import BytesIO

# config
st.set_page_config(
    page_title="Análise de Regressão - Preço",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Title
st.markdown('<h1 class="main-header">  Análise de Regressão Múltipla - Preço</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configurações")
st.sidebar.markdown("---")

# File upload
uploaded_file = st.sidebar.file_uploader(
    " Carregar arquivo Excel",
    type=['xlsx', 'xls'],
    help="Carregue seu arquivo de dados em formato Excel"
)

# Load data
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_excel(file)
        return df
    return None

# Initialize data
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("✅ Arquivo carregado com sucesso!")
else:
    st.sidebar.info("👆 Carregue um arquivo Excel para começar")
    df = None

# Main content
if df is not None:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        " Dados",
        " Análise Exploratória",
        " Correlações",
        " Modelo de Regressão",
        " Predições",
        " Exportar Resultados"
    ])
    
    with tab1:
        st.header("📋 Visão Geral dos Dados")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Registros", df.shape[0])
        with col2:
            st.metric("Total de Variáveis", df.shape[1])
        with col3:
            st.metric("Valores Nulos", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicatas", df.duplicated().sum())
        
        st.subheader(" Primeiras Linhas")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader(" Estatísticas Descritivas")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("🔍 Tipos de Dados")
        dtype_df = pd.DataFrame({
            'Coluna': df.columns,
            'Tipo': df.dtypes.values,
            'Valores Únicos': [df[col].nunique() for col in df.columns],
            'Valores Nulos': [df[col].isnull().sum() for col in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with tab2:
        st.header(" Análise Exploratória")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Distribution plots
        st.subheader(" Distribuições das Variáveis Numéricas")
        
        col1, col2 = st.columns(2)
        
        for idx, col in enumerate(numeric_cols):
            with col1 if idx % 2 == 0 else col2:
                fig = px.histogram(
                    df, 
                    x=col, 
                    marginal="box",
                    title=f"Distribuição: {col}",
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        if 'cavaco' in df.columns and 'preco' in df.columns:
            st.subheader(" Dispersão: Cavaco vs Preço")
            fig = px.scatter(
                df,
                x='cavaco',
                y='preco',
                trendline="ols",
                title="Relação entre Quantidade de Cavaco e Preço",
                labels={'cavaco': 'Quantidade de Cavaco', 'preco': 'Preço'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_traces(marker=dict(size=10, opacity=0.6))
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(" Box Plots - Detecção de Outliers")
        selected_var = st.selectbox("Selecione a variável:", numeric_cols)
        
        fig = px.box(
            df,
            y=selected_var,
            title=f"Box Plot: {selected_var}",
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header(" Análise de Correlações")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr(method='pearson')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(" Matriz de Correlação")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="YlOrRd",
                    square=True,
                    cbar_kws={"shrink": 0.8},
                    ax=ax
                )
                plt.title("Correlação de Pearson", fontsize=14, weight='bold')
                st.pyplot(fig)
            
            with col2:
                st.subheader(" Maiores Correlações")
                if 'preco' in numeric_cols:
                    target_corr = corr_matrix['preco'].drop('preco').sort_values(ascending=False)
                    
                    for var, corr in target_corr.items():
                        correlation_strength = "Forte" if abs(corr) > 0.7 else "Moderada" if abs(corr) > 0.4 else "Fraca"
                        st.metric(
                            label=var,
                            value=f"{corr:.3f}",
                            delta=correlation_strength
                        )
            
            st.subheader("Pairplot - Relações entre Variáveis")
            
            selected_vars = st.multiselect(
                "Selecione as variáveis para o pairplot:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if len(selected_vars) >= 2:
                with st.spinner("Gerando pairplot..."):
                    fig = px.scatter_matrix(
                        df[selected_vars],
                        dimensions=selected_vars,
                        title="Matriz de Dispersão",
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig.update_traces(diagonal_visible=False, showupperhalf=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Necessário pelo menos 2 variáveis numéricas para análise de correlação.")
    
    with tab4:
        st.header(" Modelo de Regressão Múltipla OLS")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuração do Modelo")
            
            target = st.selectbox(
                "Variável Dependente (Y):",
                numeric_cols,
                index=numeric_cols.index('preco') if 'preco' in numeric_cols else 0
            )
            
            available_predictors = [col for col in numeric_cols if col != target]
            
            default_predictors = ['cavaco', 'peso_peca', 'comprimento']
            default_predictors = [p for p in default_predictors if p in available_predictors]
            
            predictors = st.multiselect(
                "Variáveis Independentes (X):",
                available_predictors,
                default=default_predictors if default_predictors else available_predictors[:3]
            )
        
        with col2:
            if len(predictors) > 0:
                formula = f"{target} ~ " + " + ".join(predictors)
                
                try:
                    model = smf.ols(formula=formula, data=df).fit()
                    
                    st.subheader("  Resumo do Modelo")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("R²", f"{model.rsquared:.4f}")
                    with col_b:
                        st.metric("R² Ajustado", f"{model.rsquared_adj:.4f}")
                    with col_c:
                        st.metric("AIC", f"{model.aic:.2f}")
                    with col_d:
                        st.metric("BIC", f"{model.bic:.2f}")
                    
                    st.text(model.summary())
                    
                    st.session_state['model'] = model
                    st.session_state['formula'] = formula
                    st.session_state['predictors'] = predictors
                    st.session_state['target'] = target
                    
                    df_results = df.copy()
                    df_results[f'{target}_fitted'] = model.fittedvalues
                    df_results['residuos'] = model.resid
                    st.session_state['df_results'] = df_results
                    
                except Exception as e:
                    st.error(f" Erro ao ajustar o modelo: {str(e)}")
            else:
                st.warning(" Selecione pelo menos uma variável independente.")
        
        if 'model' in st.session_state:
            st.subheader("🔍 Diagnóstico do Modelo")
            
            model = st.session_state['model']
            
            tab_a, tab_b, tab_c = st.tabs(["Resíduos", "Q-Q Plot", "Valores Ajustados"])
            
            with tab_a:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=model.fittedvalues,
                    y=model.resid,
                    mode='markers',
                    marker=dict(color='#1f77b4', size=8, opacity=0.6),
                    name='Resíduos'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title="Resíduos vs Valores Ajustados",
                    xaxis_title="Valores Ajustados",
                    yaxis_title="Resíduos",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab_b:
                fig, ax = plt.subplots(figsize=(8, 6))
                sm.qqplot(model.resid, line='45', ax=ax)
                plt.title("Q-Q Plot dos Resíduos")
                st.pyplot(fig)
            
            with tab_c:
                target = st.session_state['target']
                
                fig = go.Figure()
                
                min_val = min(df[target].min(), model.fittedvalues.min())
                max_val = max(df[target].max(), model.fittedvalues.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Ajuste Perfeito'
                ))
                
                fig.add_trace(go.Scatter(
                    x=df[target],
                    y=model.fittedvalues,
                    mode='markers',
                    marker=dict(color='#1f77b4', size=8, opacity=0.6),
                    name='Observado vs Ajustado'
                ))
                
                fig.update_layout(
                    title="Valores Observados vs Valores Ajustados",
                    xaxis_title="Valores Observados",
                    yaxis_title="Valores Ajustados",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("  Visualização Interativa")
            
            if 'cavaco' in df.columns and target == 'preco':
                df_results = st.session_state['df_results']
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['cavaco'],
                    y=df['preco'],
                    mode='markers',
                    name='Preço Observado',
                    marker=dict(color='grey', opacity=0.6, size=7)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['cavaco'],
                    y=df_results['preco_fitted'],
                    mode='markers',
                    name='Preço Ajustado',
                    marker=dict(color='#FFA500', opacity=0.6, size=7)
                ))
                
                x = df['cavaco'].values
                y = df['preco'].values
                mask = ~np.isnan(x) & ~np.isnan(y)
                x_clean, y_clean = x[mask], y[mask]
                
                if len(x_clean) > 1:
                    coef = np.polyfit(x_clean, y_clean, 1)
                    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                    y_line = np.polyval(coef, x_line)
                    
                    fig.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        name='Regressão Linear Simples',
                        line=dict(color='#440154', width=2)
                    ))
                
                fig.update_layout(
                    title="Preço Observado vs Ajustado (Modelo Múltiplo)",
                    xaxis_title="Quantidade de Cavaco",
                    yaxis_title="Preço",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header(" Fazer Predições")
        
        if 'model' in st.session_state:
            model = st.session_state['model']
            predictors = st.session_state['predictors']
            target = st.session_state['target']
            
            st.subheader(" Insira os valores para predição")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                input_data = {}
                
                num_cols = 3
                cols = st.columns(num_cols)
                
                for idx, var in enumerate(predictors):
                    with cols[idx % num_cols]:
                        var_mean = df[var].mean()
                        var_min = float(df[var].min())
                        var_max = float(df[var].max())
                        
                        input_data[var] = st.number_input(
                            f"{var}",
                            min_value=var_min * 0.5,
                            max_value=var_max * 1.5,
                            value=var_mean,
                            step=(var_max - var_min) / 100
                        )
                
                if st.button(" Fazer Predição", type="primary"):
                    try:
                        new_data = pd.DataFrame([input_data])
                        prediction = model.predict(new_data)
                        
                        st.session_state['last_prediction'] = prediction.iloc[0]
                        st.session_state['last_input'] = input_data
                    except Exception as e:
                        st.error(f" Erro na predição: {str(e)}")
            
            with col2:
                if 'last_prediction' in st.session_state:
                    st.success(" Predição realizada!")
                    
                    st.markdown("###  Resultado")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="color: #1f77b4; margin: 0;">{st.session_state['last_prediction']:.2f}</h2>
                        <p style="margin: 0; color: #666;">Valor predito de {target}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("###   Valores de Entrada")
                    for var, value in st.session_state['last_input'].items():
                        st.write(f"**{var}:** {value:.3f}")
                    
                    try:
                        prediction_summary = model.get_prediction(pd.DataFrame([st.session_state['last_input']]))
                        conf_int = prediction_summary.conf_int(alpha=0.05)
                        
                        st.markdown("###  Intervalo de Confiança (95%)")
                        st.write(f"**Inferior:** {conf_int[0][0]:.2f}")
                        st.write(f"**Superior:** {conf_int[0][1]:.2f}")
                    except:
                        pass
            
            st.subheader("Predições em Lote")
            
            uploaded_pred_file = st.file_uploader(
                "Carregar arquivo para predições em lote (Excel)",
                type=['xlsx', 'xls'],
                key="batch_predictions"
            )
            
            if uploaded_pred_file is not None:
                try:
                    batch_df = pd.read_excel(uploaded_pred_file)
                    
                    missing_vars = [var for var in predictors if var not in batch_df.columns]
                    
                    if missing_vars:
                        st.error(f" Variáveis ausentes no arquivo: {', '.join(missing_vars)}")
                    else:
                        predictions = model.predict(batch_df[predictors])
                        batch_df[f'{target}_predito'] = predictions
                        
                        st.success(f" {len(batch_df)} predições realizadas!")
                        st.dataframe(batch_df, use_container_width=True)
                        
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            batch_df.to_excel(writer, index=False, sheet_name='Predições')
                        
                        st.download_button(
                            label="💾 Download Predições",
                            data=output.getvalue(),
                            file_name="predicoes_lote.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f" Erro ao processar arquivo: {str(e)}")
        else:
            st.warning(" Por favor, ajuste o modelo na aba 'Modelo de Regressão' primeiro.")
    
    with tab6:
        st.header(" Exportar Resultados")
        
        if 'df_results' in st.session_state:
            df_results = st.session_state['df_results']
            model = st.session_state['model']
            
            st.subheader(" Dados com Valores Ajustados e Resíduos")
            st.dataframe(df_results, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_results.to_excel(writer, index=False, sheet_name='Resultados')
                
                st.download_button(
                    label=" Download Excel - Resultados Completos",
                    data=output.getvalue(),
                    file_name="modelo_resultados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
            
            with col2:
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=" Download CSV - Resultados Completos",
                    data=csv,
                    file_name="modelo_resultados.csv",
                    mime="text/csv"
                )
            
            st.subheader(" Relatório do Modelo")
            
            report = f"""
            # Relatório de Análise de Regressão
            
            ## Configuração do Modelo
            - **Fórmula:** {st.session_state['formula']}
            - **Variável Dependente:** {st.session_state['target']}
            - **Variáveis Independentes:** {', '.join(st.session_state['predictors'])}
            - **Observações:** {model.nobs}
            
            ## Métricas de Desempenho
            - **R²:** {model.rsquared:.4f}
            - **R² Ajustado:** {model.rsquared_adj:.4f}
            - **F-statistic:** {model.fvalue:.4f}
            - **Prob (F-statistic):** {model.f_pvalue:.4e}
            - **AIC:** {model.aic:.2f}
            - **BIC:** {model.bic:.2f}
            - **Log-Likelihood:** {model.llf:.2f}
            
            ## Coeficientes
            """
            
            for param, coef in model.params.items():
                pvalue = model.pvalues[param]
                significance = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else ""
                report += f"\n- **{param}:** {coef:.4f} (p-value: {pvalue:.4e}) {significance}"
            
            report += "\n\n*Significância: *** p<0.001, ** p<0.01, * p<0.05*"
            
            st.markdown(report)
            
            st.download_button(
                label="📄 Download Relatório (Markdown)",
                data=report,
                file_name="relatorio_modelo.md",
                mime="text/markdown"
            )
        else:
            st.warning(" Por favor, ajuste o modelo na aba 'Modelo de Regressão' primeiro.")

else:
    st.markdown("""
    ## Sistema de Análise de Regressão para peças industriais
    
    Esta aplicação permite realizar análises completas de regressão múltipla de forma interativa e intuitiva.
    
    ### Funcionalidades:
    
    - **Análise Exploratória:** Visualize distribuições, outliers e estatísticas descritivas
    - **Correlações:** Análise de correlação com heatmaps e pairplots interativos
    - **Modelagem OLS:** Regressão múltipla com diagnósticos completos
    - **Predições:** Faça predições individuais ou em lote
    - **Exportação:** Baixe resultados em Excel, CSV ou relatórios em Markdown

    ### Como usar:

    1. **Carregue seus dados:** Use a barra lateral para fazer upload de um arquivo Excel
    2. **Explore:** Navegue pelas abas para analisar seus dados
    3. **Modele:** Configure e ajuste seu modelo de regressão
    4. **Prediga:** Use o modelo para fazer novas predições
    5. **Exporte:** Baixe os resultados da análise
    
    ###  Formato dos dados:
    
    Seu arquivo Excel deve conter colunas numéricas, incluindo:
    - Uma variável dependente (target) - ex: `preco`
    - Variáveis independentes (features) - ex: `cavaco`, `peso_peca`, `comprimento`
    
    ---
    
    ** Comece carregando seu arquivo na barra lateral!**
    """)
    
    st.subheader(" Exemplo de Estrutura de Dados")
    sample_data = pd.DataFrame({
        'PN': ['peça1', 'peça2', 'peça3'],
        'preco': [4351.07, 11043.78, 605.90],
        'peso_mp': [19.221, 66.762, 2.227],
        'peso_peca': [2.918, 1.247, 0.526],
        'cavaco': [16.303, 65.515, 1.701],
        'comprimento': [370, 1740, 290]
    })
    st.dataframe(sample_data, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>  Análise de Regressão v1.1</p>
        <p>Desenvolvido com Streamlit</p>
    </div>
    """, unsafe_allow_html=True)