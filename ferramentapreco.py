import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import io
from simulacaopreco import simulacao_preco
from scipy.stats import skew, norm
from sklearn.model_selection import cross_val_score

def ferramenta_preco():
    st.title('Projeto para estimar o preço de uma casa utilizando modelos de regressão')
    st.write("""
    ### Descrição do problema de previsão de preço de casas:

    O desafio é descrever a casa dos sonhos de alguém, que muitas vezes vai além de características simples como o número de quartos ou se tem uma cerca branca. O conjunto de dados de Ames, Iowa, nos EUA, contém 79 variáveis explicativas que descrevem quase todos os aspectos de casas residenciais. A meta é prever o preço final de cada casa, com a precisão sendo avaliada pelo Erro Quadrático Médio das Raízes (RMSE) entre o logaritmo do valor previsto e o logaritmo do preço de venda observado. Isso significa que previsões igualmente ponderam casas caras e baratas em termos de erros de previsão.

    Saiba mais sobre o contexto deste problema [aqui](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).

    **Reconhecimentos:**
    O dataset de Ames foi compilado por Dean De Cock para educação em ciência de dados, e é uma alternativa incrível para cientistas de dados em busca de uma versão moderna e expandida do conhecido conjunto de dados de habitação de Boston.

    **Objetivo:**
    Prever o preço de venda de cada casa, com cada identificação no conjunto de teste, prevendo a variável `SalePrice`.

    """)
    st.markdown('<p style="font-size:24px;">Simule os preços das casas de Ames aqui!</p>', unsafe_allow_html=True)
    # Exibindo a ferramenta de simulação
    Ferramenta_Simulacao = simulacao_preco()
    st.write("Quer descobrir como foi feita essa ferramenta? Me acompanhe nos próximos passos!")
    # Exiba o código por trás da previsão
    st.markdown('<p style="font-size:24px;">Parte 1: Primeiras impressões dos dados</p>', unsafe_allow_html=True)
    st.write("Primeiro, vamos ter uma visão inicial dos nossos dados de treino:")

    with st.echo():
        #bring in the six packs
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        train.head()
    st.dataframe(train.head(),hide_index=True)

    with st.echo():
        train.describe()
    st.dataframe(train.describe())
    st.write("Visualizando as estatísticas descritivas do conjunto de dados, pode-se ter uma noção inicial de medidas como média, desvio padrão, mínimo, 25º percentil, mediana (50º percentil), 75º percentil e máximo para cada coluna numérica. Essa observação vai ser importante para manipulações nessas variáveis numéricas no futuro do projeto.")

    with st.echo():
        train.info()
    buffer = io.StringIO()
    train.info(buf=buffer)
    traininfo = buffer.getvalue()
    st.text(traininfo)

    st.write("Em uma primeira análise, percebe-se que temos algumas variáveis númericas e categóricas com valores nulos, algo que será tratado no pré processamento dos dados.")

    st.markdown('<p style="font-size:24px;">Parte 2: Pré processamento dos dados</p>', unsafe_allow_html=True)
    with st.echo():
        matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
        prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
        prices.hist()
        #log transform the target:
        train["SalePrice"] = np.log1p(train["SalePrice"])
    fig, ax = plt.subplots()
    prices.hist(ax=ax)
    st.pyplot(fig)
    st.write("Aqui, estamos visualizando a distribuição dos preços das casas antes e depois de aplicar uma transformação logarítmica. A transformação logarítmica é comumente usada para normalizar a distribuição de características númericas que são altamente assimétricas. A vantagem de ter características com distribuição normal é que muitos modelos estatísticos assumem essa distribuição para os dados e, portanto, é bastante importante na aplicação de modelos de regressão que vamos utilizar.")

    st.markdown('<p style="font-size:20px;">Parte 2.1: Juntando os dados de treino com os de teste e normalizando as variáveis numéricas</p>', unsafe_allow_html=True)
    with st.echo():
        # Removendo as colunas com muitos valores nulos
        train.drop(['PoolQC', 'MiscFeature', 'Fence', 'Alley'], axis=1, inplace=True)
        test.drop(['PoolQC', 'MiscFeature', 'Fence', 'Alley'], axis=1, inplace=True)
        all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                        test.loc[:,'MSSubClass':'SaleCondition']))
        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
        #log transform skewed numeric features:
        skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
        skewed_feats = skewed_feats[skewed_feats > 0.75]
        skewed_feats = skewed_feats.index

        all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    st.write("""
    Neste trecho do código, estamos realizando algumas transformações importantes nos dados:

    1. **Remoção de colunas com muitos valoresn nulos**: Primeiro, removemos colunas que têm muitos valores faltantes, como `PoolQC`, `MiscFeature`,`Fence` e `Alley`.

    2. **Junção dos dados de treino com os dados de teste**: Como é? Qual é o sentido de juntar os dados de treino com os dados de teste? Simples! Nos próximos passos vamos fazer diversas transformações nos dados e é importantes que essas transformações sejam feitas tanto nos dados de treino quanto nos dados de teste. Mas não se preocupe! Ainda vamos separar esses dados novamente.

    3. **Identificação de características numéricas assimétricas**: Primeiro, identificamos todas as características numéricas do conjunto de dados. Em seguida, calculamos a assimetria (skewness) de cada uma dessas características. Se a assimetria de uma característica for maior que 0,75, consideramos essa característica como assimétrica.

    4. **Transformação logarítmica de características assimétricas**: As características identificadas como assimétricas são então transformadas usando a função `log1p`.

    Esta transformação é crucial para muitos modelos de Machine Learning, pois eles podem se comportar melhor quando as características têm uma distribuição mais normalizada.
    """)

    st.markdown('<p style="font-size:20px;">Parte 2.2: Fazendo o tratamento das variáveis categóricas</p>', unsafe_allow_html=True)
    with st.echo():
        # Preencher os valores nulos das variáveis categóricas com a moda (o valor mais frequente)
        categorical_cols = all_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            all_data[col].fillna(all_data[col].mode()[0], inplace=True)

    # Explicação do que o código faz
    st.write("""
        Nesta parte, estamos preenchemos os valores ausentes nas colunas categóricas com a moda dessas colunas, ou seja, o valor mais comum. 
        Isso é feito para evitar que nosso modelo seja afetado por dados faltantes, o que pode prejudicar seu desempenho.
    """)

    st.markdown('<p style="font-size:20px;">Parte 2.3: Criando dummies das variáveis categóricas</p>', unsafe_allow_html=True)
    with st.echo():
        all_data = pd.get_dummies(all_data)

    st.write("""
    Neste trecho do código, estamos realizando uma etapa essencial de pré-processamento chamada **codificação one-hot**:

    - **Codificação One-Hot**: Convertendo variáveis categóricas em um formato que pode ser fornecido aos algoritmos de Machine Learning para fazer uma melhor predição. Ao usar a função `get_dummies`, o pandas criará uma coluna numérica para cada categoria possível em cada variável categórica original. Cada uma dessas colunas conterá 1 se a categoria original tiver o valor correspondente e 0 caso contrário.

    Esta etapa é crucial porque muitos modelos de aprendizado de máquina não conseguem lidar diretamente com variáveis categóricas. A codificação one-hot permite que você transforme essas variáveis categóricas em um formato binário que os modelos podem entender e usar.
    """)

    st.markdown('<p style="font-size:20px;">Parte 2.4: Fazendo o tratamento das variáveis numéricas</p>', unsafe_allow_html=True)
    with st.echo():
        # filling NA's with the mean of the column:
        all_data = all_data.fillna(all_data.mean())

    st.write("""
    Neste trecho do código, estamos lidando com valores faltantes (NA's) nas nossas colunas:

    - **Preenchendo NA's com a média**: Estamos usando a função `fillna` do pandas para substituir todos os valores faltantes (NA's) pela média da respectiva coluna. 

    O tratamento de valores faltantes é uma etapa fundamental no pré-processamento de dados. Deixar valores faltantes pode resultar em erros durante a modelagem ou em previsões imprecisas. Ao substituir os valores faltantes pela média da coluna, estamos utilizando uma abordagem simples e comum para lidar com esses valores ausentes sem introduzir um viés excessivo.
    """)

    st.markdown('<p style="font-size:20px;">Parte 2.5: Preparação dos dados para modelagem </p>', unsafe_allow_html=True)
    with st.echo():
        # creating matrices for sklearn:
        X_train = all_data[:train.shape[0]]
        X_test = all_data[train.shape[0]:]
        y = train.SalePrice


    st.write("""
    Neste trecho do código, estamos preparando os dados para o processo de modelagem e separando os dados de treino e dados de teste (Eu avisei para não se preocupar!):

    - **X_train**: Esta é a matriz de recursos para o conjunto de treinamento. Estamos pegando todas as linhas do dataset `all_data` até o tamanho de `train`, o que corresponde aos dados de treinamento.

    - **X_test**: Esta é a matriz de recursos para o conjunto de teste. Começa exatamente de onde o `X_train` termina, ou seja, após o último registro do conjunto de treinamento em `all_data` e vai até o final, representando assim os dados de teste.

    - **y**: Este é o vetor alvo, representando os preços das casas. Como estamos prevendo o preço das casas, extraímos a coluna `SalePrice` do dataset `train` para ser nossa variável alvo.

    Essa separação é essencial para treinar o modelo usando `X_train` e `y` e, em seguida, fazer previsões no conjunto `X_test`.
    """)

    st.markdown('<p style="font-size:24px;">Parte 3: Modelos de Regressão</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:16px;">Antes de avançarmos, é importante entender os três modelos de regressão que testaremos: Regressão Linear, Ridge e Lasso.</p>

    - **Regressão Linear**: É o modelo de regressão mais básico que procura estabelecer uma relação linear entre as variáveis independentes (features) e a variável dependente. Não possui qualquer tipo de regularização.

    - **Regressão Ridge**: É uma extensão da regressão linear que inclui uma penalidade L2. Esta penalidade tem o efeito de "encolher" os coeficientes para evitar o overfitting. O grau de penalização é controlado pelo parâmetro alpha: um alpha de 0 é equivalente à regressão linear normal, enquanto um alpha muito grande faria com que todos os coeficientes se aproximassem de zero, resultando em um modelo muito simples.

    - **Regressão Lasso**: Semelhante à regressão Ridge, mas a penalidade é L1. O diferencial do Lasso é que ele pode reduzir alguns coeficientes exatamente a zero, o que é útil se acreditarmos que muitos features são irrelevantes ou redundantes.

    A métrica que usaremos para testar a eficácia desses modelos é o Root Mean Squared Error (RMSE). O RMSE é uma métrica comumente usada em tarefas de regressão e mede a quantidade de erro que existe entre dois conjuntos de dados. Em outras palavras, ele nos diz o quanto nossa previsão desvia, em média, dos valores reais. 

    Menores valores de RMSE indicam um ajuste melhor ao modelo, enquanto valores mais altos indicam um ajuste pior. No entanto, é importante notar que o RMSE não tem uma escala fixa, por isso é melhor usar RMSE em comparação com outras medidas do mesmo conjunto de dados ao invés de usar de forma absoluta.
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:20px;">Parte 3.1: Função da Raiz do Erro Quadrático Médio (RMSE) </p>', unsafe_allow_html=True)
    with st.echo():
        def rmse_cv(model):
            rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5))
            return(rmse)

    st.write("""
    Neste trecho de código, estamos definindo uma função chamada `rmse_cv`:

    - **Função RMSE**: A função `rmse_cv` é projetada para calcular a Raiz do Erro Quadrático Médio (Root Mean Squared Error - RMSE). Você pode estar se perguntando onde estão os dados de teste na definição dessa função e isso está relacionado com a validação cruzada. Essa técnica divide o conjunto de treinamento em 5 subconjuntos e o modelo é treinado 5 vezes, cada vez deixando de fora um dos subconjuntos para ser o teste e o restante sendo usado para treinamento. A média dos erros em cada um desses treinos traz uma estimativa do desempenho do modelo, por isso nós só temos dados de treino nos argumentos dessa função! Lembre-se que quanto menor o RMSE, melhor o modelo em termos de precisão.

    A função retornará o RMSE para o modelo fornecido, ajudando-nos a comparar e escolher o melhor modelo para o conjunto de dados.
    """)

    st.markdown('<p style="font-size:20px;">Parte 3.2: Regressão Linear </p>', unsafe_allow_html=True)
    with st.echo():
        from sklearn.linear_model import LinearRegression

        model_linear = LinearRegression()
        rmse_linear = rmse_cv(model_linear).mean()

    st.write("""
    Neste segmento, estamos realizando as seguintes operações:

    - **Linear Regression**: Utilizando a regressão linear do Scikit-learn. Esta é a regressão básica sem qualquer tipo de regularização.

    - **rmse_linear**: Calculamos o RMSE usando validação cruzada para o modelo de regressão linear simples.

    Dado que não temos um hiperparâmetro para sintonizar como nas regressões Ridge e Lasso, a operação é mais direta.
    """)

    with st.echo():
        print(f"O RMSE para a regressão linear é {rmse_linear:.5f}.")

    st.write(f"O RMSE para a regressão linear é {rmse_linear:.5f}.")

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

    - **Gráfico**: Plotamos os resultados em um gráfico para visualizar o RMSE em relação aos diferentes valores de alpha. O objetivo é encontrar um valor ótimo de alpha que minimize o RMSE mas acabamos encontrando o logo da Nike hahaha.
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
        print(f"O valor mínimo de RMSE, para a regressão de Lasso, é {min_rmse_lasso:.4f} e ocorre quando alpha é {min_alpha_lasso:.5f}.")

    st.write(f"O valor mínimo de RMSE, para a regressão de Lasso, é {min_rmse_lasso:.4f} e ocorre quando alpha é {min_alpha_lasso:.5f}.")
    st.write("""
    Portanto, temos o nosso modelo com menor RMSE: **Lasso!**
    """)

    st.markdown('<p style="font-size:24px;">Parte 4: Modelo de Regressão de Lasso</p>', unsafe_allow_html=True)
    st.write("Conforme dito anteriormente, a principal diferença entre a Regressão de Lasso e a de Ridge é que na de Lasso podemos reduzir os seus coeficientes a zero, ou seja, eliminar variáveis! Vamos dar uma olhada na quantidade de variáveis que foram eliminadas e nas variáveis mais importantes de acordo com esse modelo.")
    st.markdown('<p style="font-size:20px;">Parte 4.1: Variáveis Mais Importantes Conforme o Modelo de Lasso </p>', unsafe_allow_html=True)
    with st.echo():
        # Obter os coeficientes do modelo Lasso
        coef = pd.Series(model_lasso.coef_, index=X_train.columns)
        st.write("Lasso selecionou " + str(sum(coef != 0)) + " variáveis e eliminou as outras " +  str(sum(coef == 0)) + " variáveis. É importante dizer que o número de variáveis é maior que o valor inicial por causa das variáveis dummies que foram criadas.")

        # Selecionar as top e bottom 10 variáveis
        imp_coef = pd.concat([coef.sort_values().head(10),
                            coef.sort_values().tail(10)])

        # Plotar os coeficientes
        fig, ax = plt.subplots(figsize=(8, 10))
        imp_coef.plot(kind='barh', ax=ax, title='Coeficientes no Modelo Lasso')
        st.pyplot(fig)
    st.write("""

    O gráfico exibe os coeficientes mais influentes do modelo Lasso para a predição dos preços das casas. Os coeficientes que tem '_' são categorias das variáveis principais, por exemplo: Neighborhood_StoneBr é a categoria StoneBr da variável Neighborhood. Estes coeficientes sugerem as seguintes interpretações:

    - **GrLivArea**: A área habitável acima do solo é o fator mais significativo, indicando que quanto maior a área habitável, maior o impacto no preço da casa.
    - **Neighborhood**: Várias categorias de bairros como StoneBr, Crawford, NoRidge e NridgHt estão entre as variáveis mais influentes, mostrando que a localização é um determinante crucial do preço.
    - **Quality and Condition Features**: Variáveis como `KitchenQual_Ex` (qualidade excelente da cozinha) e `OverallQual` (qualidade geral da casa) são fortes preditores, destacando a importância da qualidade e condição geral.
    - **LotArea**: O tamanho do lote também é uma variável importante, refletindo a preferência por propriedades com mais espaço ao ar livre.
    - **Type of Sale and Condition**: `SaleType_WD` (venda de garantia devedora) e `SaleCondition_Abnorml` (condições anormais de venda) têm um peso significativo, o que pode ser interpretado como a influência de diferentes tipos de venda e condições na determinação dos preços.
    - **MSZoning**: As zonas de classificação como `MSZoning_RM` (zona residencial de média densidade) têm coeficientes negativos, sugerindo que casas nesses locais podem ser avaliadas a preços mais baixos em comparação com outras categorias de zonamento.
    - **OverallQual**: A qualidade geral da casa, representada por `OverallQual`, é um indicador-chave. Coeficientes positivos significativos para essa variável mostram que casas de qualidade superior tendem a alcançar preços mais altos.
    - **RoofMatl**: O material do telhado, especialmente `RoofMatl_ClyTile` (telhas de argila), tem um coeficiente negativo considerável, o que pode indicar que telhados com esse material podem não ser preferidos ou podem estar associados a características de casas que não são valorizadas no mercado atual.

    Guarde essa análise! Vamos nos basear nela para construir a simulação dos preços das casas no Streamlit.""")

    st.markdown('<p style="font-size:20px;">Parte 4.2: Submissão dos Resultados Finais no Kaggle! </p>', unsafe_allow_html=True)
    st.write("Finalmente! Vamos fazer previsões com o modelo de Lasso só que dessa vez usando a base de dados de teste. Como transformamos a variável de destino usando `log1p`, devemos aplicar `expm1` às previsões para reverter essa transformação.")

    with st.echo():
        # Fazer previsões com o modelo de Lasso
        lasso_preds = model_lasso.predict(X_test)

        # Retransformar as previsões com np.expm1
        lasso_preds_expm = np.expm1(lasso_preds)

    st.write("Por fim, vamos criar um DataFrame com essas previsões e salvá-lo como um arquivo CSV para submeter na competição do Kaggle.")

    with st.echo():
        # Criar um DataFrame com as previsões
        predictions_df = pd.DataFrame()
        predictions_df['Id'] = test['Id']  
        predictions_df['SalePrice'] = lasso_preds_expm
        predictions_df.head()

        # Salvar as previsões em um arquivo CSV
        predictions_df.to_csv('lasso_predictions.csv', index=False)

    # Mostrar o DataFrame no aplicativo
    st.write(predictions_df.head())

    # Adicionar um botão para baixar as previsões como CSV
    st.write('Vamos baixar as previsões como um arquivo CSV para submeter no Kaggle:')
    st.download_button(label='Download CSV', data=predictions_df.to_csv(index=False), file_name='lasso_predictions.csv', mime='text/csv')
    # Texto com link incorporado para a competição do Kaggle
    # Adicionar uma imagem ao Streamlit
    st.image('Leaderboard_House.png', caption='Leaderboard do Kaggle')
    st.markdown("Sucesso! Conseguimos ficar no top 15% da [competição no Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/leaderboard).")
    st.markdown('<p style="font-size:24px;">Parte 5: Simulação dos Preços no Streamlit</p>', unsafe_allow_html=True)
    st.write("Ok, mas você deve estar se perguntando como que a simulação dos preços das casas foi feito no Streamlit? Bom, uma vez que temos apenas 5 variáveis para o usuário escolher, torna-se necessário refazer o processo para o contexto dessas 5 variáveis. É importante ressaltar que essas variáveis foram escolhidas conforme os resultados de importância dos coeficientes de Lasso! Ah, é válido dizer que teremos um desempenho pior nesse modelo em comparação com o original por causa do menor volume de dados.")
    with st.echo():
        # Concatene os dados de treino e teste mantendo apenas as 5 variáveis de interesse e a variável alvo 'SalePrice'
        train_5_variaveis_data = train[['GrLivArea', 'LotArea', 'MSZoning', 'Neighborhood', 'OverallQual', 'SalePrice']]
        test_5_variaveis_data = test[['GrLivArea', 'LotArea', 'MSZoning', 'Neighborhood', 'OverallQual']]
        all_data_5variaveis = pd.concat((train_5_variaveis_data,
                                        test_5_variaveis_data))

        # Encontrar as características numéricas
        numeric_feats_5variaveis = all_data_5variaveis.dtypes[all_data_5variaveis.dtypes != "object"].index

        # Computar a assimetria (skewness) e realizar a transformação de log nas features com assimetria maior que 0.75
        skewed_feats_5variaveis = all_data_5variaveis[numeric_feats_5variaveis].apply(lambda x: skew(x.dropna()))
        skewed_feats_5variaveis = skewed_feats_5variaveis[skewed_feats_5variaveis > 0.75]
        skewed_feats_5variaveis = skewed_feats_5variaveis.index

        all_data_5variaveis[skewed_feats_5variaveis] = np.log1p(all_data_5variaveis[skewed_feats_5variaveis])

        # Aplicar one-hot encoding nas variáveis categóricas
        all_data_5variaveis = pd.get_dummies(all_data_5variaveis)

        # Garantir que as colunas de treino e teste sejam as mesmas
        final_train, final_test = all_data_5variaveis[:train_5_variaveis_data.shape[0]], all_data_5variaveis[train_5_variaveis_data.shape[0]:]

        # Alinhar as colunas no conjunto de treino e teste
        final_train, final_test = final_train.align(final_test, join='left', axis=1, fill_value=0)
        final_test.drop(['SalePrice'], axis=1, inplace=True)

        # Agora final_train e final_test têm as mesmas colunas, então podemos treinar o modelo e fazer previsões
        alphas_5variaveis = [10, 5, 1, 0.1, 0.001, 0.0005]
        model_lasso_5variaves = LassoCV(alphas=alphas_5variaveis, max_iter=100000).fit(final_train.drop(['SalePrice'], axis=1), (final_train['SalePrice']))
        lasso_preds_5variaveis = model_lasso_5variaves.predict(final_test)

        # Retransformar as previsões com np.expm1
        lasso_preds_expm_5variaveis = np.expm1(lasso_preds_5variaveis)

        predictions_df_5variaveis = pd.DataFrame()
        predictions_df_5variaveis['Id'] = test['Id']  
        predictions_df_5variaveis['SalePrice'] = lasso_preds_expm_5variaveis

        # Salvar as previsões em um arquivo CSV
        predictions_df_5variaveis.to_csv('lasso_predictions_5variaveis.csv', index=False)

    # Adicionar um botão para baixar as previsões como CSV
    st.write('Vamos baixar as previsões como um arquivo CSV para submeter no Kaggle:')
    st.download_button(label='Download CSV', data=predictions_df_5variaveis.to_csv(index=False), file_name='lasso_predictions_5variaveis.csv', mime='text/csv')
    st.image('5variaveis_score.png', caption='Score da previsão')
    st.write('Conforme esperado, tivemos um desempenho pior do que o modelo que tinha todas as variáveis mas ainda está ok. É o suficiente para nós termos uma boa estimativa dos preços das casas na nossa ferramenta do Streamlit!')




