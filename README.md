# 📊 Aplicação de Análise de Regressão Múltipla - Streamlit

Esta aplicação Streamlit permite realizar análises completas de regressão múltipla de forma interativa, baseada no seu notebook Jupyter original.

## 🚀 Funcionalidades

### 1. 📋 Análise de Dados
- Visualização de dados brutos
- Estatísticas descritivas completas
- Detecção de valores nulos e duplicatas
- Informações sobre tipos de dados

### 2. 📈 Análise Exploratória
- Histogramas com box plots para todas as variáveis numéricas
- Gráficos de dispersão (scatter plots) interativos
- Box plots para detecção de outliers
- Análise da relação Cavaco vs Preço

### 3. 🔗 Análise de Correlações
- Matriz de correlação com heatmap
- Identificação das maiores correlações com a variável alvo
- Pairplot interativo (matriz de dispersão)
- Análise de Pearson

### 4. 🎯 Modelo de Regressão OLS
- Seleção flexível de variáveis dependentes e independentes
- Ajuste automático do modelo OLS
- Métricas de desempenho (R², R² Ajustado, AIC, BIC)
- Resumo estatístico completo do modelo
- **Diagnósticos:**
  - Gráfico de resíduos vs valores ajustados
  - Q-Q Plot para normalidade dos resíduos
  - Valores observados vs valores ajustados
  - Visualização interativa com Plotly

### 5. 🔮 Predições
- **Predições individuais:** Interface intuitiva para entrada de valores
- **Predições em lote:** Upload de arquivo Excel para múltiplas predições
- Intervalos de confiança (95%)
- Download dos resultados em Excel

### 6. 💾 Exportação de Resultados
- Download de dados com valores ajustados e resíduos
- Exportação em Excel e CSV
- Relatório completo do modelo em Markdown
- Estatísticas e coeficientes formatados

## 📦 Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passos de Instalação

1. **Clone ou baixe os arquivos do projeto**

2. **Crie um ambiente virtual (recomendado)**
```bash
python -m venv venv

# No Windows:
venv\Scripts\activate

# No Linux/Mac:
source venv/bin/activate
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

## 🎯 Como Usar

### Iniciar a Aplicação

```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no seu navegador em `http://localhost:8501`

### Passo a Passo

1. **Upload dos Dados**
   - Na barra lateral, clique em "Carregar arquivo Excel"
   - Selecione seu arquivo `.xlsx` ou `.xls`
   - O arquivo deve conter suas variáveis numéricas

2. **Exploração dos Dados**
   - Navegue pela aba "📋 Dados" para ver estatísticas e informações gerais
   - Vá para "📈 Análise Exploratória" para visualizar distribuições e outliers

3. **Análise de Correlações**
   - Na aba "🔗 Correlações", examine o heatmap
   - Visualize as relações entre variáveis no pairplot

4. **Ajuste do Modelo**
   - Na aba "🎯 Modelo de Regressão":
     - Selecione a variável dependente (ex: `preco`)
     - Selecione as variáveis independentes (ex: `cavaco`, `peso_peca`, `comprimento`)
   - O modelo será ajustado automaticamente
   - Analise as métricas e diagnósticos

5. **Fazer Predições**
   - Na aba "🔮 Predições":
     - **Individual:** Insira valores manualmente e clique em "Fazer Predição"
     - **Lote:** Faça upload de um arquivo Excel com novos dados

6. **Exportar Resultados**
   - Na aba "💾 Exportar Resultados":
     - Baixe os dados completos com valores ajustados
     - Baixe o relatório do modelo

## 📊 Estrutura dos Dados

Seu arquivo Excel deve ter a seguinte estrutura:

| PN    | preco      | peso_mp | peso_peca | cavaco  | comprimento |
|-------|------------|---------|-----------|---------|-------------|
| peça1 | 4351.07    | 19.221  | 2.918     | 16.303  | 370         |
| peça2 | 11043.78   | 66.762  | 1.247     | 65.515  | 1740        |
| peça3 | 605.90     | 2.227   | 0.526     | 1.701   | 290         |

**Observações:**
- A primeira coluna pode ser um identificador (texto)
- As demais colunas devem ser numéricas
- Valores nulos serão tratados automaticamente

## 🎨 Características Técnicas

### Bibliotecas Utilizadas
- **Streamlit:** Interface web interativa
- **Pandas:** Manipulação de dados
- **NumPy:** Operações numéricas
- **Plotly:** Visualizações interativas
- **Seaborn/Matplotlib:** Gráficos estatísticos
- **Statsmodels:** Modelagem estatística (OLS)
- **OpenPyXL:** Leitura/escrita de arquivos Excel

### Modelo de Regressão
- **Tipo:** Regressão Linear Múltipla (OLS)
- **Método:** Ordinary Least Squares
- **Diagnósticos:** Resíduos, normalidade, homocedasticidade

## 🔧 Solução de Problemas

### Erro ao carregar arquivo
- Verifique se o arquivo é `.xlsx` ou `.xls`
- Confirme que há colunas numéricas no arquivo
- Verifique se não há erros de formatação no Excel

### Erro ao ajustar modelo
- Certifique-se de ter pelo menos uma variável independente
- Verifique se as variáveis selecionadas são numéricas
- Confira se não há muitos valores nulos

### Predições incorretas
- Confirme que o modelo foi ajustado com sucesso
- Verifique se as variáveis de entrada estão corretas
- Para lote, certifique-se que o arquivo tem as mesmas colunas

## 📝 Exemplo de Uso

```python
# Dados de exemplo para predição individual:
cavaco: 5.96
peso_peca: 0.719
comprimento: 853

# Resultado esperado:
preco predito: ~2500 (variará conforme seu modelo)
```

## 🤝 Suporte
a
Para questões ou problemas:
1. Verifique se todas as dependências foram instaladas
2. Confirme que está usando Python 3.8+
3. Revise a estrutura dos seus dados

## 📄 Licença

Este projeto é fornecido como está, para uso educacional e comercial.

---

**Desenvolvido com ❤️ usando Streamlit**

Versão: 1.0  
Última atualização: 2025