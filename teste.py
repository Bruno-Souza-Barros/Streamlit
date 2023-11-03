import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from scipy.stats import skew, norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


#bring in the six packs
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
st.title('Dados de preço de casa')
st.write('Com 79 variáveis ​​explicativas que descrevem muitos aspectos das residências em Ames, Iowa, esta base de dados foi uma ótima experiência para aprender sobre EDA (Análise exploratória de dados).')

# Exiba o código por trás da previsão
st.subheader('Código por trás desta previsão:')
st.markdown('<p style="font-size:24px;">Parte 1: Primeiras impressões dos dados</p>', unsafe_allow_html=True)
st.write("Primeiro, vamos ter uma visão inicial dos nossos dados:")

with st.echo():
    train.head()
st.dataframe(train.head(),hide_index=True)

with st.echo():
    train.describe()
st.dataframe(train.describe())
st.write("Visualizando as estatísticas descritivas do conjunto de dados, pode-se ter uma noção inicial de medidas como média, desvio padrão, mínimo, 25º percentil, mediana (50º percentil), 75º percentil e máximo para cada coluna numérica. Essa observação vai ser importante para manipulações nessas variáveis numéricas no futuro.")

st.markdown('<p style="font-size:24px;">Parte 2: Pré processamento dos dados</p>', unsafe_allow_html=True)
with st.echo():
    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
    prices.hist()
fig, ax = plt.subplots()
prices.hist(ax=ax)
st.pyplot(fig)
st.write("Aqui, estamos visualizando a distribuição dos preços das casas antes e depois de aplicar uma transformação logarítmica. A transformação logarítmica é comumente usada para normalizar a distribuição de características númericas que são altamente assimétricas. A vantagem de ter características com distribuição normal é que muitos modelos estatísticos assumem essa distribuição para os dados e, portanto, é bastante importante na aplicação de modelos de regressão que vamos utilizar.")

st.markdown('<p style="font-size:20px;">Parte 2.1: Normalizando as variáveis numéricas</p>', unsafe_allow_html=True)
with st.echo():
    #log transform the target:
    train["SalePrice"] = np.log1p(train["SalePrice"])

    #log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

st.write("""
Neste trecho do código, estamos realizando algumas transformações importantes nos dados:

1. **Transformação logarítmica do preço das casas**: A variável alvo, `SalePrice`, é transformada usando a função `log1p` que aplica log(1+x) a todos os elementos da coluna. Isso é útil para lidar com características numéricas que são assimétricas.

2. **Identificação de características numéricas assimétricas**: Primeiro, identificamos todas as características numéricas do conjunto de dados. Em seguida, calculamos a assimetria (skewness) de cada uma dessas características. Se a assimetria de uma característica for maior que 0,75, consideramos essa característica como assimétrica.

3. **Transformação logarítmica de características assimétricas**: As características identificadas como assimétricas são então transformadas usando a função `log1p`.

Esta transformação é crucial para muitos modelos de Machine Learning, pois eles podem se comportar melhor quando as características têm uma distribuição mais normalizada.
""")

st.markdown('<p style="font-size:20px;">Parte 2.2: Criando dummies das variáveis categóricas</p>', unsafe_allow_html=True)
with st.echo():
    all_data = pd.get_dummies(all_data)

st.write("""
Neste trecho do código, estamos realizando uma etapa essencial de pré-processamento chamada **codificação one-hot**:

- **Codificação One-Hot**: Convertendo variáveis categóricas em um formato que pode ser fornecido aos algoritmos de Machine Learning para fazer uma melhor predição. Ao usar a função `get_dummies`, o pandas criará uma coluna numérica para cada categoria possível em cada variável categórica original. Cada uma dessas colunas conterá 1 se a categoria original tiver o valor correspondente e 0 caso contrário.

Esta etapa é crucial porque muitos modelos de aprendizado de máquina não conseguem lidar diretamente com variáveis categóricas. A codificação one-hot permite que você transforme essas variáveis categóricas em um formato binário que os modelos podem entender e usar.
""")

st.markdown('<p style="font-size:20px;">Parte 2.3: Preenchendo valores númericos faltantes com a média da variável</p>', unsafe_allow_html=True)
with st.echo():
    # filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())

st.write("""
Neste trecho do código, estamos lidando com valores faltantes (NA's) nas nossas colunas:

- **Preenchendo NA's com a média**: Estamos usando a função `fillna` do pandas para substituir todos os valores faltantes (NA's) pela média da respectiva coluna. 

O tratamento de valores faltantes é uma etapa fundamental no pré-processamento de dados. Deixar valores faltantes pode resultar em erros durante a modelagem ou em previsões imprecisas. Ao substituir os valores faltantes pela média da coluna, estamos utilizando uma abordagem simples e comum para lidar com esses valores ausentes sem introduzir um viés excessivo.
""")

st.markdown('<p style="font-size:20px;">Parte 2.4: Preparação dos dados para modelagem </p>', unsafe_allow_html=True)
with st.echo():
    # creating matrices for sklearn:
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice


st.write("""
Neste trecho do código, estamos preparando os dados para o processo de modelagem:

- **X_train**: Esta é a matriz de recursos para o conjunto de treinamento. Estamos pegando todas as linhas do dataset `all_data` até o tamanho de `train`, o que corresponde aos dados de treinamento.

- **X_test**: Esta é a matriz de recursos para o conjunto de teste. Começa exatamente de onde o `X_train` termina, ou seja, após o último registro do conjunto de treinamento em `all_data` e vai até o final, representando assim os dados de teste.

- **y**: Este é o vetor alvo, representando os preços das casas. Como estamos prevendo o preço das casas, extraímos a coluna `SalePrice` do dataset `train` para ser nossa variável alvo.

Essa separação é essencial para treinar o modelo usando `X_train` e `y` e, em seguida, fazer previsões no conjunto `X_test`.
""")

st.markdown('<p style="font-size:24px;">Parte 3: Modelos de Regressão</p>', unsafe_allow_html=True)
st.markdown("""
<p style="font-size:20px;">Antes de avançarmos, é importante entender os três modelos de regressão linear que testaremos: Regressão Linear Normal, Ridge e Lasso.</p>

- **Regressão Linear Normal**: É o modelo de regressão mais básico que procura estabelecer uma relação linear entre as variáveis independentes (features) e a variável dependente. Não possui qualquer tipo de regularização.

- **Regressão Ridge**: É uma extensão da regressão linear que inclui uma penalidade L2. Esta penalidade tem o efeito de "encolher" os coeficientes para evitar o overfitting. O grau de penalização é controlado pelo parâmetro alpha: um alpha de 0 é equivalente à regressão linear normal, enquanto um alpha muito grande faria com que todos os coeficientes se aproximassem de zero, resultando em um modelo muito simples.

- **Regressão Lasso**: Semelhante à regressão Ridge, mas a penalidade é L1. O diferencial do Lasso é que ele pode reduzir alguns coeficientes exatamente a zero, o que é útil se acreditarmos que muitos features são irrelevantes ou redundantes.

A métrica que usaremos para testar a eficácia desses modelos é o Root Mean Squared Error (RMSE). O RMSE é uma métrica comumente usada em tarefas de regressão e mede a quantidade de erro que existe entre duas conjuntos de dados. Em outras palavras, ele nos diz o quanto nossa previsão desvia, em média, dos valores reais. A fórmula para calcular o RMSE é:

Menores valores de RMSE indicam um ajuste melhor ao modelo, enquanto valores mais altos indicam um ajuste pior. No entanto, é importante notar que o RMSE não tem uma escala fixa, por isso é melhor usar RMSE em comparação com outras medidas do mesmo conjunto de dados ao invés de usar de forma absoluta.
""", unsafe_allow_html=True)

st.markdown('<p style="font-size:20px;">Parte 3.1: Função da Raiz do Erro Quadrático Médio (RMSE) </p>', unsafe_allow_html=True)
with st.echo():
    def rmse_cv(model):
        rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5))
        return(rmse)

st.write("""
Neste trecho de código, estamos definindo uma função chamada `rmse_cv`:

- **Função**: A função `rmse_cv` é projetada para calcular a Raiz do Erro Quadrático Médio (Root Mean Squared Error - RMSE) usando validação cruzada. É uma métrica comum para medir a precisão de um modelo em tarefas de regressão. Quanto menor o RMSE, melhor o modelo em termos de precisão.

A função retornará o RMSE para o modelo fornecido, ajudando-nos a comparar e escolher o melhor modelo para o conjunto de dados.
""")

st.markdown('<p style="font-size:20px;">Parte 3.2: Regressão Linear Simples </p>', unsafe_allow_html=True)
with st.echo():
    from sklearn.linear_model import LinearRegression

    model_linear = LinearRegression()
    rmse_linear = rmse_cv(model_linear).mean()

st.write("""
Neste segmento, estamos realizando as seguintes operações:

- **Linear Regression**: Utilizando a regressão linear simples do Scikit-learn. Esta é a regressão básica sem qualquer tipo de regularização.

- **rmse_linear**: Calculamos o RMSE usando validação cruzada para o modelo de regressão linear simples.

Dado que não temos um hiperparâmetro para sintonizar como nas regressões Ridge e Lasso, a operação é mais direta.
""")

with st.echo():
    print(f"O RMSE para a regressão linear simples é {rmse_linear:.5f}.")

st.write(f"O RMSE para a regressão linear simples é {rmse_linear:.5f}.")

st.markdown('<p style="font-size:20px;">Parte 3.3: Regressão de Ridge </p>', unsafe_allow_html=True)
with st.echo():
    from sklearn.linear_model import Ridge
    
    model_ridge = Ridge()
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
    cv_ridge = pd.Series(cv_ridge, index=alphas)
    fig, ax = plt.subplots()
    cv_ridge.plot(title="Relação do RMSE com o Alpha (Ridge)", ax=ax)
    plt.xlabel("alpha")
    plt.ylabel("rmse")

st.pyplot(fig)

st.write("""
Neste trecho, estamos realizando a seguinte operação:

- **Ridge Regression**: Estamos usando a regressão Ridge do Scikit-learn. Este é um tipo de regressão linear regularizada que inclui uma penalidade L2.

- **Alphas**: São os valores que determinam a força da regularização em Ridge Regression. Valores maiores de alpha oferecem mais regularização e podem prevenir overfitting.

- **cv_ridge**: Aqui, calculamos o RMSE usando validação cruzada para cada valor de alpha usando a função `rmse_cv` que definimos anteriormente.

- **Gráfico**: Plotamos os resultados em um gráfico para visualizar o RMSE em relação aos diferentes valores de alpha. O objetivo é encontrar um valor ótimo de alpha que minimize o RMSE.
""")

with st.echo():
    min_alpha, min_rmse = cv_ridge.idxmin(), cv_ridge.min()
    print(f"O valor mínimo de RMSE é {min_rmse:.5f} e ocorre quando alpha é {min_alpha}.")

min_alpha, min_rmse = cv_ridge.idxmin(), cv_ridge.min()
st.write(f"O valor mínimo de RMSE, para a regressão de Ridge, é {min_rmse:.5f} e ocorre quando alpha é {min_alpha}.")

st.markdown('<p style="font-size:20px;">Parte 3.4: Regressão de Lasso </p>', unsafe_allow_html=True)

# Treinar o modelo Lasso usando LassoCV
with st.echo():
    from sklearn.linear_model import LassoCV
    
    # Definindo os alphas que você deseja testar
    alphas = [1, 0.1, 0.001, 0.0005]
    
    model_lasso = LassoCV(alphas=alphas, max_iter=100000).fit(X_train, y)
    cv_lasso = pd.Series(model_lasso.mse_path_.mean(axis=-1), index=model_lasso.alphas_)
    
    # Plotando o RMSE em relação ao alpha
    fig, ax = plt.subplots()
    cv_lasso.plot(title="Relação do RMSE com o Alpha (Lasso).", ax=ax)
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    st.pyplot(fig)

st.write("""
Neste segmento, estamos realizando as operações:

- **Lasso Regression com CV**: Utilizamos a classe LassoCV do Scikit-learn que faz validação cruzada para encontrar o melhor alpha.

- **Alphas**: São os valores que determinam a força da regularização. Valores maiores de alpha resultarão em mais regularização.

- **Gráfico**: Os resultados são plotados em um gráfico para visualizar o RMSE em relação aos diferentes valores de alpha. 
""")

with st.echo():
    min_alpha_lasso = model_lasso.alpha_
    min_rmse_lasso = np.sqrt(min(cv_lasso))
    print(f"O valor mínimo de RMSE, para a regressão de Lasso, é {min_rmse_lasso:.5f} e ocorre quando alpha é {min_alpha_lasso:.5f}.")

st.write(f"O valor mínimo de RMSE, para a regressão de Lasso, é {min_rmse_lasso:.5f} e ocorre quando alpha é {min_alpha_lasso:.5f}.")

st.markdown('<p style="font-size:24px;">Parte 4: Modelo de Regressão de Lasso</p>', unsafe_allow_html=True)

with st.echo():
    # Obter os coeficientes do modelo Lasso
    coef = pd.Series(model_lasso.coef_, index=X_train.columns)
    st.write("Lasso selecionou " + str(sum(coef != 0)) + " variáveis e eliminou as outras " +  str(sum(coef == 0)) + " variáveis")

    # Selecionar as top e bottom 10 variáveis
    imp_coef = pd.concat([coef.sort_values().head(10),
                          coef.sort_values().tail(10)])

    # Plotar os coeficientes
    fig, ax = plt.subplots(figsize=(8, 10))
    imp_coef.plot(kind='barh', ax=ax, title='Coeficientes no Modelo Lasso')
    st.pyplot(fig)

st.title('Previsão de Preço de Casas com Regressão Lasso')

# Coletando as entradas do usuário
GrLivArea = st.slider('Área acima do solo (em pés quadrados):', min_value=500, max_value=4000)
LotArea = st.slider('Tamanho do lote (em pés quadrados):', min_value=1000, max_value=20000)

MSZoning = st.selectbox('Classificação de Zoneamento:', ['Agricultura', 'Comercial', 'Residencial em Vilas Flutuantes', 'Industrial', 
                                                        'Residencial de Alta Densidade', 'Residencial de Baixa Densidade', 
                                                        'Parque Residencial de Baixa Densidade', 'Residencial de Média Densidade'])

Neighborhood = st.selectbox('Bairro:', ['Alturas de Bloomington', 'Bluestem', 'Briardale', 'Brookside', 
                                        'Clear Creek', 'College Creek', 'Crawford', 'Edwards', 'Gilbert', 
                                        'Departamento de Transportes de Iowa e Ferrovia', 'Vila Meadow', 
                                        'Mitchell', 'Ames Norte', 'Northridge', 'Villa Northpark', 'Alturas Northridge', 
                                        'Ames Noroeste', 'Cidade Velha', 'Sul e Oeste da Universidade Estadual de Iowa', 
                                        'Sawyer', 'Oeste de Sawyer', 'Somerset', 'Stone Brook', 'Timberland', 'Veenker'])

OverallQual = st.selectbox('Qualidade Geral:', ['Muito Excelente', 'Excelente', 'Muito Bom', 'Bom', 
                                               'Acima da Média', 'Média', 'Abaixo da Média', 'Razoável', 'Fraco', 'Muito Fraco'])

# Transformação reversa das entradas para corresponder à base de dados
MSZoning_dict = {
    'Agricultura': 'A',
    'Comercial': 'C',
    'Residencial em Vilas Flutuantes': 'FV',
    'Industrial': 'I',
    'Residencial de Alta Densidade': 'RH',
    'Residencial de Baixa Densidade': 'RL',
    'Parque Residencial de Baixa Densidade': 'RP',
    'Residencial de Média Densidade': 'RM'
}

Neighborhood_dict = {
    'Alturas de Bloomington': 'Blmngtn',
    'Bluestem': 'Blueste'
    # ... continue para todos os bairros
}

OverallQual_dict = {
    'Muito Excelente': 10,
    'Excelente': 9
    # ... continue para todas as qualidades
}

MSZoning = MSZoning_dict[MSZoning]
Neighborhood = Neighborhood_dict[Neighborhood]
OverallQual = OverallQual_dict[OverallQual]

# Criar o dataframe de entrada
input_data = pd.DataFrame({
    'GrLivArea': [GrLivArea],
    'LotArea': [LotArea],
    'MSZoning': [MSZoning],
    'Neighborhood': [Neighborhood],
    'OverallQual': [OverallQual]
})

# Previsão usando o modelo
predicted_price = model_lasso.predict(input_data)

# Mostrar previsão
st.write(f'Previsão de Preço: ${predicted_price[0]:,.2f}')

# Rodar o app com: streamlit run your_script_name.py




