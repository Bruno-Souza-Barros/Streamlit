import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import io

@st.cache_data(show_spinner=False)
def simulacao_churn():
    st.title("Projeto para classificar quais clientes de um banco podem dar churn")
    st.write("""
    ### Descrição do Problema de Churn de Clientes:

    Um gerente do banco está preocupado com o número crescente de clientes deixando os serviços de cartão de crédito. Seria muito útil se pudéssemos prever quem vai deixar o serviço para que possam proativamente oferecer melhores serviços aos clientes e influenciar sua decisão na direção oposta.

    Este conjunto de dados foi obtido do site [Analyttica](https://leaps.analyttica.com/home). 

    O dataset consiste em 10.000 clientes mencionando sua idade, salário, estado civil, limite do cartão de crédito, categoria do cartão de crédito, entre outros. Há aproximadamente 19 características.

    Temos apenas 16.07% dos clientes que cancelaram seus serviços. Assim, é um pouco difícil treinar nosso modelo para prever quais clientes estão propensos ao churn.
    """)

    # Carregar o dataset
    df = pd.read_csv('BankChurners.csv')

    # Manter apenas as colunas relevantes
    columns_of_interest = ['Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Relationship_Count', 
                        'Months_Inactive_12_mon', 'Total_Trans_Amt', 'Contacts_Count_12_mon', 
                        'Attrition_Flag']
    df = df[columns_of_interest]
    df['Attrition_Flag'] = df['Attrition_Flag'].replace({'Attrited Customer': 1, 'Existing Customer': 0})

    # Separar em características e alvo
    X = df.drop('Attrition_Flag', axis=1)
    y = df['Attrition_Flag']

    # Obter valores mínimos e máximos
    min_values = X.min()
    max_values = X.max()

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar os dados
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar o modelo
    model = XGBClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    return min_values,max_values,model,X,scaler

@st.cache_data(show_spinner=False)
def ferramenta_churn():
    st.markdown('<p style="font-size:24px;">Parte 1: Primeiras impressões dos dados</p>', unsafe_allow_html=True)
    with st.echo():
        # Carregando o conjunto de dados
        bank_churners = pd.read_csv('BankChurners.csv')

        # Retirando algumas colunas desnecessárias
        bank_churners_df = bank_churners.iloc[:,1:-2]

        # Exibindo as primeiras linhas do dataframe
        st.dataframe(bank_churners_df.head())


    with st.echo():
        bank_churners_df.info()

    col1, col2 = st.columns(2)
    with col1:
        buffer = io.StringIO()
        bank_churners_df.info(buf=buffer)
        bank_churners_df_info = buffer.getvalue()
        st.text(bank_churners_df_info)
        st.write("Primeiramente, percebe-se que temos variáveis númericas e categóricas, porém nenhuma delas com valor nulo.")
        st.write("Além disso, temos 19 variáveis independes que vão nos ajudar a fazer a previsão da variável dependente 'Attrition_Flag'.")
        st.write("Todas as colunas da nossa base de dados estão descritas ao lado. Em uma primeira análise, nota-se variáveis tanto pessoais, como idade e nível educacional, e variáveis financeiras, tal qual 'limite de crédito'.")

    with col2:
        st.markdown("""
    <small>
    <b>Attrition_Flag</b>: Indica se o cliente deixou de ser cliente do banco (Churn) ou não. "Existing Customer" significa que o cliente ainda está com o banco, enquanto "Attritted Customer" significa que o cliente deu Churn.<br>
    <small>
    <b>Customer_Age</b>: Idade do cliente.<br>
    <small>
    <b>Gender</b>: Gênero do cliente (M para masculino, F para feminino).<br>
    <small>
    <b>Dependent_count</b>: Número de dependentes que o cliente possui.<br>
    <small>
    <b>Education_Level</b>: Nível educacional do cliente (por exemplo, ensino médio, graduação etc.).<br>
    <small>
    <b>Marital_Status</b>: Estado civil do cliente (casado, solteiro etc.).<br>
    <small>
    <b>Income_Category</b>: Categoria de renda anual do cliente (por exemplo, "$60K - $80K", "Menos de $40K" etc.).<br>
    <small>
    <b>Card_Category</b>: Categoria do cartão de crédito (Blue, Silver, Gold, Platinum).<br>
    <small>
    <b>Months_on_book</b>: Número de meses desde que o cliente está com o banco.<br>
    <small>
    <b>Total_Relationship_Count</b>: Número total de produtos que o cliente tem com o banco.<br>
    <small>
    <b>Months_Inactive_12_mon</b>: Número de meses em que o cliente esteve inativo nos últimos 12 meses.<br>
    <small>
    <b>Contacts_Count_12_mon</b>: Número de vezes que o cliente entrou em contato com o banco nos últimos 12 meses.<br>
    <small>
    <b>Credit_Limit</b>: Limite de crédito no cartão de crédito.<br>
    <small>
    <b>Total_Revolving_Bal</b>: Saldo total rotativo no cartão de crédito.<br>
    <small>
    <b>Avg_Open_To_Buy</b>: Média de crédito não utilizado nos últimos 12 meses.<br>
    <small>
    <b>Total_Amt_Chng_Q4_Q1</b>: Mudança na quantidade transacionada entre o quarto e o primeiro trimestre.<br>
    <small>
    <b>Total_Trans_Amt</b>: Montante total das transações nos últimos 12 meses.<br>
    <small>
    <b>Total_Trans_Ct</b>: Contagem total de transações nos últimos 12 meses.<br>
    <small>
    <b>Total_Ct_Chng_Q4_Q1</b>: Mudança na contagem de transações entre o quarto e o primeiro trimestre.<br>
    <small>
    <b>Avg_Utilization_Ratio</b>: Razão média de utilização do cartão de crédito.<br>
    </small>
    """, unsafe_allow_html=True)


    # Título da parte
    st.markdown('<p style="font-size:24px;">Parte 2: Análise Exploratória dos Dados</p>', unsafe_allow_html=True)

    st.markdown('<p style="font-size:20px;">Parte 2.1: Análise das Variáveis Categóricas </p>', unsafe_allow_html=True)
    
    # Código por trás dos gráficos, apresentado com st.echo
    with st.echo():
        # Definindo as colunas de informações pessoais e financeiras
        category_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
        
        # Função para criar e exibir o gráfico
        def create_plot(column_name):
            # Contagem de clientes por categoria e Attrition_Flag
            category_counts = bank_churners_df.groupby([column_name, 'Attrition_Flag']).size().reset_index(name='count')
            # Calculando o total para cada categoria
            total_counts = bank_churners_df[column_name].value_counts().reset_index(name='total')
            # Merge para adicionar a coluna total no dataframe de contagens
            category_counts = category_counts.merge(total_counts, on=column_name)
            # Calculando a porcentagem para cada categoria dentro de Attrition_Flag
            category_counts['percentage'] = (category_counts['count'] / category_counts['total'] * 100).round().astype(int)
            
            # Criando o gráfico de barras com Plotly
            fig = px.bar(category_counts, x=column_name, y='count', color='Attrition_Flag',
                        title=f'Distribuição por {column_name}', text='percentage')
            
            # Atualizando os textos das barras para percentuais sem parênteses
            fig.update_traces(texttemplate='%{text}%')

            # Adicionando anotações para o total acima da barra completa
            for cat in category_counts[column_name].unique():
                total = category_counts[category_counts[column_name] == cat]['count'].sum()
                fig.add_annotation(x=cat, y=total, text=str(total),
                                showarrow=False, yshift=10, font=dict(color='black'))
            
            # Atualizando o layout para remover grade e título do eixo Y, e adicionando margens
            fig.update_layout(showlegend=True, yaxis_showgrid=False, yaxis_title='', yaxis_showticklabels=False, plot_bgcolor='white',
                            margin=dict(l=0, r=0, t=40, b=30))
            
            return fig

        # Gerando os gráficos para cada coluna categórica
        for col in category_cols:
            fig = create_plot(col)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - **Gênero (Gender)**: A distribuição de gênero mostra duas categorias distintas com uma predominância do gênero feminino, que representa 53% das ocorrências no conjunto de dados. Todavia, pode-se dizer que, nesta categoria, temos uma distribuição uniforme.

    - **Nível de Educação (Education_Level)**: Com 7 categorias únicas, a categoria 'Graduate' se destaca como a mais comum, abrangendo 3.128 clientes. Este dado pode refletir um foco do banco em clientes com formação superior ou pode indicar uma correlação entre nível de educação e utilização de serviços bancários.

    - **Estado Civil (Marital_Status)**: Entre as 4 categorias de estado civil, 'Married' (casado) é o mais frequente com 4.687 casos. Isso pode sugerir que pessoas casadas tendem a ter maior engajamento com serviços financeiros e que o banco tenha um foco no segmento demográfico de adultos. Vamos explorar a distribuição dos clientes por idade no próximo tópico!
                
    - **Categoria de Renda (Income_Category)**: Há 6 faixas de renda diferentes sendo representadas, com a maioria dos clientes (53%) situando-se na categoria 'Less than $60K'. Percebe-se, também, que esse banco atende várias faixas de renda.

    - **Categoria do Cartão (Card_Category)**: Entre as categorias de cartões oferecidos, a categoria 'Blue' é de longe a mais comum, com 9.436 ocorrências. Isso pode refletir uma estratégia de marketing bem-sucedida, uma preferência dos clientes por este tipo de cartão ou limitações dos tipos 'Silver', 'Gold' e 'Platinum'.

        """)

    st.markdown('<p style="font-size:20px;">Parte 2.2: Análise das Variáveis Numéricas </p>', unsafe_allow_html=True)
    # Criando histogramas com Plotly
    with st.echo():
    # Função para criar e exibir o histograma com percentuais corretos
    # Selecionar as colunas numéricas do dataframe
        num_cols = bank_churners_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Cores para as categorias do Attrition_Flag conforme a segunda imagem
        color_map = {'Existing Customer': '#ADD8E6', 'Attrited Customer': '#00008B'}

        # Função para remover outliers
        def remove_outliers(df, column_name):
            Q3 = df[column_name].quantile(0.99)
            Q1 = df[column_name].quantile(0.01)
            return df[(df[column_name] <= Q3) & (df[column_name] >= Q1)]

        # Função para criar e exibir o gráfico KDE
        def create_kde_plot(df, column_name):
            df = remove_outliers(df, column_name)
            fig = px.histogram(df, x=column_name, color='Attrition_Flag',
                            marginal='violin',  # ou 'density' para KDE
                            histnorm='density',  # Normalizar para densidade
                            barmode='overlay',  # Sobrepor para KDE
                            color_discrete_map=color_map,  # Mapear cores
                            nbins=30)  # Número de bins pode ser ajustado conforme necessário
            fig.update_traces(opacity=0.75)  # Ajustar transparência
            fig.update_layout(title=f'Distribuição de {column_name} por Attrition_Flag',
                            xaxis_title=column_name, yaxis_title='',
                            plot_bgcolor='white', legend_title_text='Attrition Flag')
            return fig

        # Exibir os gráficos KDE para cada variável numérica
        for col in num_cols:
            if col not in ['CLIENTNUM']:  # Excluindo a coluna de identificação do cliente
                fig = create_kde_plot(bank_churners_df, col)
                st.plotly_chart(fig, use_container_width=True)

    st.write("""
    Os gráficos acima apresentam duas formas de visualização de dados sobrepostas: um histograma e um gráfico de violino. Ambos são utilizados para mostrar a distribuição das variáveis numéricas em relação a situação de permanência ou saída do banco (Attrition_Flag). Vamos analisar em detalhes cada uma dessas visualizações:

    ##### Gráfico de Violino
    - O gráfico de violino é uma combinação de um diagrama de caixa (box plot) e um gráfico de densidade de probabilidade (KDE - Kernel Density Estimate). Esta combinação fornece uma visualização rica e informativa que permite observar não apenas as métricas resumidas, como mediana e quartis (do box plot), mas também a forma completa da distribuição dos dados (do KDE).
    - **Parte Superior (Azul Escuro)**: Representa os clientes que deixaram o banco (_Attritted Customer_). A largura do violino em diferentes alturas indica a densidade de clientes nessa faixa etária que deram churn. Uma maior largura significa uma maior concentração de clientes que saíram do banco.
    - **Parte Inferior (Azul Claro)**: Mostra os clientes que permaneceram com o banco (_Existing Customer_). A interpretação da densidade é a mesma aqui.

    ##### Histograma
    - O histograma, localizado abaixo do gráfico de violino, exibe a contagem de clientes em diferentes faixas etárias, separados por cor conforme a situação de churn. Isso nos ajuda a ver quantitativamente a distribuição da idade dos clientes e a comparar as diferenças entre os grupos.

    A sobreposição desses dois métodos de visualização proporciona uma visão abrangente que destaca não apenas as diferenças quantitativas entre os clientes que permanecem e os que saem, mas também as tendências e padrões subjacentes na idade dos clientes que podem estar associados ao churn.
    """)
    st.write("""
    ##### Análises das Variáveis que mais Chamaram Atenção
    - **Total_Revolving_Bal**: Há uma distinção clara nas distribuições dos saldos rotativos entre as duas categorias. Clientes que permaneceram com o banco tendem a ter saldos rotativos mais altos, sugerindo que clientes mais engajados com o uso de seu crédito são mais propensos a ficar.

    - **Total_Trans_Amt**: A distribuição da quantidade total transacionada nos últimos 12 meses também mostra diferenças significativas. Clientes que deixaram o banco tendem a ter valores de transação mais baixos, o que pode indicar menos atividade e engajamento com os serviços do banco.

    - **Total_Trans_Ct**: O número total de transações segue um padrão semelhante ao da quantidade total transacionada, com aqueles que deixaram o banco tendo menos transações no geral. Isso reforça a ideia de que a atividade do cliente é um fator chave na retenção de clientes.

    - **Avg_Utilization_Ratio**: O gráfico de distribuição da razão média de utilização do cartão de crédito mostra que os clientes que deram churn geralmente têm uma utilização menor de seus cartões de crédito, sugerindo que a utilização do cartão pode ser um indicador de lealdade ao banco.

    """)


    st.markdown('<p style="font-size:20px;">Parte 2.3: Correlação entre as Variáveis Numéricas </p>', unsafe_allow_html=True)
    with st.echo():
        # Substitua os valores na coluna 'Attrition_Flag'
        bank_churners_df['Attrition_Flag'] = bank_churners_df['Attrition_Flag'].replace({'Attrited Customer': 1, 'Existing Customer': 0})

        # Selecione apenas colunas numéricas
        df_numerico = bank_churners_df.select_dtypes(include=['int64', 'float64'])

        # Calcule a matriz de correlação
        heat = df_numerico.corr()
        fig, ax = plt.subplots(figsize=[16, 8])
        sns.heatmap(heat, annot=True, ax=ax)
        st.pyplot(fig)
    st.markdown("""
    A matriz de correlação nos permite identificar como diferentes variáveis numéricas estão relacionadas com a probabilidade de um cliente deixar o banco. Aqui estão os principais insights em relação a `Attrition_Flag`:

    - **Relacionamento Total**: A variável `Total_Relationship_Count` mostra uma correlação negativa com `Attrition_Flag`. Isso sugere que clientes com um número maior de produtos ou serviços do banco tendem a permanecer com o banco, indicando que um relacionamento mais profundo com o banco pode levar a uma maior retenção de clientes.

    - **Inatividade**: Existe uma correlação positiva entre `Months_Inactive_12_mon` e `Attrition_Flag`, o que implica que clientes que permaneceram inativos por um período mais longo nos últimos 12 meses têm maior probabilidade de encerrar suas contas. Isso pode ser um indicador importante para o banco intervir e reengajar com esses clientes antes que eles decidam sair.

    - **Contatos com o Cliente**: A variável `Contacts_Count_12_mon` tem correlação positiva com `Attrition_Flag`. Clientes que entraram em contato com o banco com mais frequência nos últimos 12 meses podem ter maior probabilidade de churn. Isso pode refletir problemas ou insatisfações que levam a um aumento na comunicação, afinal ninguém vai querer ligar para um banco se realmente não precisar, certo? hahaha.

    - **Saldo Rotativo**: `Total_Revolving_Bal` possui uma correlação negativa com `Attrition_Flag`. Isso pode indicar que clientes com um saldo rotativo mais alto em seus cartões de crédito tendem a ser mais retidos pelo banco. Saldo rotativo mais alto pode estar associado a um uso mais regular do cartão de crédito, o que pode ser um sinal de engajamento do cliente com os serviços do banco.

    - **Utilização do Cartão de Crédito**: `Avg_Utilization_Ratio` tem uma correlação negativa com `Attrition_Flag`. Clientes com maior taxa de utilização de seus limites de crédito podem ser menos propensos a deixar o banco, o que pode sinalizar uma dependência contínua e satisfação com os serviços de crédito oferecidos.

        """)

    st.markdown('<p style="font-size:24px;">Parte 3: Pré processamento dos dados</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:20px;">Parte 3.1: Criando dummies das variáveis categóricas </p>', unsafe_allow_html=True)
    with st.echo():
        bank_churners_df.Gender = bank_churners_df.Gender.replace({'F':1,'M':0})
        bank_churners_df = pd.concat([bank_churners_df,pd.get_dummies(bank_churners_df['Education_Level']).drop(columns=['Unknown'])],axis=1)
        bank_churners_df = pd.concat([bank_churners_df,pd.get_dummies(bank_churners_df['Income_Category']).drop(columns=['Unknown'])],axis=1)
        bank_churners_df = pd.concat([bank_churners_df,pd.get_dummies(bank_churners_df['Marital_Status']).drop(columns=['Unknown'])],axis=1)
        bank_churners_df = pd.concat([bank_churners_df,pd.get_dummies(bank_churners_df['Card_Category']).drop(columns=['Platinum'])],axis=1)
        bank_churners_df.drop(columns = ['Education_Level','Income_Category','Marital_Status','Card_Category'],inplace=True)
        
    st.write("""
    Neste trecho do código, estamos realizando uma etapa essencial de pré-processamento chamada **codificação one-hot**:

    - **Codificação One-Hot**: Convertendo variáveis categóricas em um formato que pode ser fornecido aos algoritmos de Machine Learning para fazer uma melhor predição. Ao usar a função `get_dummies`, o pandas criará uma coluna numérica para cada categoria possível em cada variável categórica original. Cada uma dessas colunas conterá 1 se a categoria original tiver o valor correspondente e 0 caso contrário.
    """)

    st.markdown('<p style="font-size:20px;">Parte 3.2: Preparação dos dados para modelagem </p>', unsafe_allow_html=True)
    # Colocando o código de normalização dentro de um st.echo
    with st.echo():

        # Separação do conjunto de dados
        X = bank_churners_df.drop('Attrition_Flag', axis=1)
        y = bank_churners_df['Attrition_Flag']

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=70)

    # Escrevendo a explicação no Streamlit
    st.write("""

    No código acima,temos:
    - `y = bank_churners_df.pop("Attrition_Flag")`: A coluna `Attrition_Flag` é separada do dataframe e atribuída a `y`. Esta será a variável que queremos prever, também conhecida como variável dependente ou variável de resposta.
    - `X = bank_churners_df`: Após remover a coluna `Attrition_Flag`, o restante do dataframe compõe o conjunto de características, `X`, que o modelo usará para fazer as previsões.

    Usamos a função `train_test_split` do scikit-learn para:
    - Dividir `X` e `y` em conjuntos de treino (`X_train`, `y_train`) e de teste (`X_test`, `y_test`).
    - A opção `train_size=0.8` indica que 80% dos dados serão usados para treino e os restantes 20% para teste.
    - `random_state=70` é usado para garantir que a divisão seja reproduzível; ou seja, sempre que o código for executado, a mesma divisão será feita, o que é útil para fins de depuração e comparação entre modelos.

    """)

    st.markdown('<p style="font-size:20px;">Parte 3.3: Normalizando as Variáveis Numéricas </p>', unsafe_allow_html=True)

    with st.echo():
        # Normalizing the data
        req_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                    'Avg_Utilization_Ratio']

        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # Escrevendo a explicação no Streamlit
    st.write("""

    Neste código, usamos `MinMaxScaler` do scikit-learn para escalar todas as variáveis numéricas para um intervalo entre 0 e 1. O `MinMaxScaler` funciona subtraindo o valor mínimo de cada variável e dividindo pelo intervalo (valor máximo - valor mínimo). Isso transforma todos os recursos de modo que estejam dentro do intervalo especificado.

    Este método é particularmente útil quando não sabemos a distribuição dos dados ou quando sabemos que a distribuição não é Gaussiana (Normal). É também a escolha padrão quando precisamos de valores normalizados dentro de um intervalo limitado.
    """)
    st.markdown('<p style="font-size:20px;">Parte 3.4: Balanceamento de Classe com SMOTE </p>', unsafe_allow_html=True)

    with st.echo():
            
        # Balanceamento com SMOTE
        smote = SMOTE(random_state=70)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

    # Colocando o código SMOTE na coluna 1
    col1, col2 = st.columns(2)
    with col1:
        with st.echo():
            # Contagem das classes após SMOTE
            smote_counts = y.value_counts().rename(index={0: 'Existing Customer', 1: 'Attrited Customer'})

            # Criando o gráfico
            fig = px.pie(smote_counts, values=smote_counts.values, names=smote_counts.index, 
                        title='Proporção antes do SMOTE',
                        color_discrete_map={'Existing Customer': '#ADD8E6', 'Attrited Customer': '#00008B'})
            
            # Transformando o gráfico de pizza em gráfico de rosca
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")
            
            # Exibindo o gráfico
            st.plotly_chart(fig, use_container_width=True)

    # Gerando e exibindo o gráfico de rosca com a distribuição balanceada na coluna 2
    with col2:
        with st.echo():
            # Contagem das classes após SMOTE
            smote_counts = y_train_sm.value_counts().rename(index={0: 'Existing Customer', 1: 'Attrited Customer'})
            
            # Criando o gráfico
            fig = px.pie(smote_counts, values=smote_counts.values, names=smote_counts.index, 
                        title='Proporção depois do SMOTE',
                        color_discrete_map={'Existing Customer': '#ADD8E6', 'Attrited Customer': '#00008B'})
            
            # Transformando o gráfico de pizza em gráfico de rosca
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")
            
            # Exibindo o gráfico
            st.plotly_chart(fig, use_container_width=True)

    st.write("""

    O Synthetic Minority Over-sampling Technique, ou SMOTE, é uma técnica de pré-processamento que visa equilibrar a distribuição de classes em conjuntos de dados desequilibrados. Em contextos de aprendizado de máquina, um conjunto de dados desequilibrado pode prejudicar o desempenho do modelo, pois tende a favorecer a classe majoritária.

    SMOTE funciona criando exemplos sintéticos da classe minoritária ao invés de criar cópias simples. Ele seleciona registros próximos no espaço de características e calcula pontos intermediários para gerar novos exemplos. Isso ajuda a superar o problema de overfitting que pode surgir com o oversampling simples, pois os novos exemplos são variações dos existentes, e não cópias exatas.

    Os gráficos apresentados ilustram o efeito do SMOTE no conjunto de dados 'BankChurners':

    - **Proporção antes do SMOTE**: Mostra a distribuição original da variável `Attrition_Flag`, onde a classe 'Existing Customer' é a dominante, representando 83.9% do total, e 'Attrited Customer' representa 16.1%. Este desequilíbrio pode levar a um modelo de aprendizado de máquina que não é sensível à classe minoritária.

    - **Proporção depois do SMOTE**: Após a aplicação do SMOTE, a nova distribuição é mostrada, onde ambas as classes estão agora igualmente representadas, cada uma com 50%. Isso indica que o conjunto de dados agora está balanceado, e o modelo de aprendizado de máquina resultante pode ser mais eficaz na captura das nuances de ambas as classes.

    """)

    st.markdown('<p style="font-size:24px;">Parte 4: Modelos de Classificação e Métricas de Avaliação de Modelo</p>', unsafe_allow_html=True)

    st.write("""
    #### Modelos de Classificação
            
    No contexto de previsão de churn de clientes, vários modelos de machine learning podem ser aplicados para identificar padrões e indicadores que contribuem para a saída de clientes. Aqui estão quatro modelos que vamos utilizar:

    ###### Regressão Logística
    - A 'Regressão Logística' é um modelo estatístico que, apesar do nome, é utilizado para classificação binária. Ele estima a probabilidade de um evento ocorrer, como o churn de um cliente, e é particularmente útil quando a relação entre a variável dependente e as variáveis independentes é logisticamente distribuída. Este modelo é robusto a pequenas amostras e variações nos dados.

    ###### Árvore da Decisão
    - A 'Árvore da Decisão' é um modelo de aprendizado supervisionado que utiliza uma estrutura de árvore para tomar decisões. Ela divide um conjunto de dados em subconjuntos menores enquanto, ao mesmo tempo, desenvolve uma árvore de decisão associada. As decisões são tomadas seguindo os caminhos da árvore até as folhas, que representam a classificação ou decisão final. Este modelo é intuitivo e fácil de visualizar, mas pode ser propenso a overfitting.

    ###### SVM (Máquina de Vetores de Suporte)
    - 'SVM' ou Máquina de Vetores de Suporte é um modelo poderoso e versátil de aprendizado supervisionado usado para classificação e regressão. Em termos de classificação, o SVM procura encontrar o hiperplano que melhor divide o conjunto de dados em classes. É particularmente eficaz em espaços de alta dimensão e em situações onde a clareza da divisão entre as classes é distinta.

    ###### XGBClassifier
    - 'XGBClassifier' refere-se a um modelo baseado em 'eXtreme Gradient Boosting'. Este é um algoritmo de aprendizado de máquina que utiliza a estratégia de boosting, combinando o desempenho de vários modelos mais fracos para criar um modelo mais forte e preciso. Ele é conhecido por sua velocidade e desempenho, especialmente em conjuntos de dados grandes e complexos, e tem opções para regularização para evitar overfitting.

    """)

    st.write("""
    #### Métricas de Avaliação de Modelo

    Ao avaliar o desempenho de modelos de machine learning, sobretudo para classificação, utilizamos várias métricas para entender como o modelo está se saindo em diferentes aspectos. Aqui estão algumas das métricas chave:

    ###### Acurácia
    - A 'Acurácia' é a métrica mais intuitiva e representa a proporção de previsões corretas (tanto verdadeiros positivos quanto verdadeiros negativos) em relação ao total de previsões feitas. Enquanto uma métrica útil, ela pode ser enganosa em conjuntos de dados desbalanceados. Por exemplo, se um modelo prever que nenhum cliente vai dar churn (quando 83% dos clientes de fato não dão churn), ele terá uma acurácia de 83% sem realmente ter aprendido qualquer padrão significativo.

    ###### Precisão
    - A 'Precisão' mede a proporção de verdadeiros positivos (clientes que deram churn e foram corretamente identificados pelo modelo) em relação ao total de previsões positivas (todos os clientes que o modelo previu que dariam churn). Esta métrica é crucial quando o custo de um falso positivo é alto. Em um conjunto de dados onde a maioria dos clientes não dá churn, um modelo que sempre prevê 'não churn' pode parecer preciso, mas na verdade não é capaz de identificar corretamente os casos de churn.

    ###### Revocação (Recall)
    - A 'Revocação' avalia a proporção de verdadeiros positivos em relação ao total de casos positivos reais (a soma de verdadeiros positivos e falsos negativos). É uma métrica importante quando é crucial capturar todos os casos positivos. Por exemplo, se estamos tentando identificar todos os clientes em risco de churn, a revocação nos diz qual proporção desses clientes realmente em risco foi capturada pelo modelo.

    ###### F1-Score
    - O 'F1-Score' é a média harmônica da Precisão e Revocação, e fornece um equilíbrio entre as duas métricas. Ele é particularmente útil quando precisamos de um balanço entre a precisão e a revocação, e é uma boa escolha em conjuntos de dados com uma distribuição de classe desigual.

    ###### Matriz de Confusão
    - A 'Matriz de Confusão' é uma tabela que permite visualizar o desempenho de um algoritmo de classificação. Ela mostra as contagens de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos. Essa matriz fornece uma visão mais detalhada do desempenho do modelo além da mera acurácia, permitindo-nos entender o tipo de erros que o modelo está cometendo.

    """)
    st.markdown('<p style="font-size:20px;">Parte 4.1: Modelos de Classificação e Métricas de Avaliaçãor</p>', unsafe_allow_html=True)
    with st.echo():
    # Treinamento e avaliação dos modelos
        models = {
            'Regressão Logística': LogisticRegression(random_state=70),
            'Árvore da Decisão': DecisionTreeClassifier(random_state=70),
            'SVM': SVC(random_state=70),
            'XGBClassifier':XGBClassifier(random_state=70, use_label_encoder=False, n_jobs=-1)
        }

        for name, model in models.items():
            st.subheader(name)
            model.fit(X_train_sm, y_train_sm)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            col1, col2, col3, col4 = st.columns(4)
            # Exibição das métricas
            with col1:
                st.write(f'Acurárcia: {acc:.2f}')
            with col2:
                st.write(f'Precisão: {prec:.2f}')
            with col3:
                st.write(f'Revocação: {recall:.2f}')
            with col4:
                st.write(f'F1-Score: {f1:.2f}')

            # Exibição da matriz de confusão
            fig = ff.create_annotated_heatmap(
                z=conf_matrix, 
                x=['Not Churn', 'Churn'], 
                y=['Not Churn', 'Churn'],
                colorscale='Viridis',
                showscale=True
            )
            fig.update_layout(
                title=f'Matriz de Confusão para {name}',
                xaxis=dict(title='Previsto'),
                yaxis=dict(title='Real')
            )
            st.plotly_chart(fig, use_container_width=True)
    st.write("Temos o nosso modelo com melhores resultados em todas as métricas: XGBClassifier! ")
    st.markdown('<p style="font-size:20px;">Parte 4.2: Aprofundamento do XGBClassifier</p>', unsafe_allow_html=True)
    with st.echo():
        xgb_model = XGBClassifier(random_state=70, use_label_encoder=False, n_jobs=-1)
        xgb_model.fit(X_train_sm, y_train_sm)
        # Obtendo as importâncias das características
        feature_importances = xgb_model.feature_importances_
        # Criando uma Series no pandas para facilitar a ordenação e visualização
        features = pd.Series(feature_importances, index=X.columns)
        # Ordenando as características pela importância
        top_features = features.sort_values(ascending=False).head(6)

        # Criando um gráfico de barras para as importâncias das características
        fig = px.bar(top_features, y=top_features.values, x=top_features.index, text=top_features.values)
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(
            title='Top 6 Variáveis Mais Importantes do XGBClassifier',
            xaxis_title='',
            yaxis_title='',
            showlegend=False,
            yaxis_showgrid=False,
            yaxis_showticklabels=False,
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    st.write("Ótimo, temos nossas 6 variáveis mais importantes conforme o modelo XGBClassifier. O interessante é que todas essas variáveis já foram exploradas antes na análise exploratória de dados.")
    st.markdown('<p style="font-size:24px;">Parte 5: Simulação de Churn no Streamlit</p>', unsafe_allow_html=True)
    st.write("Ok, mas você deve estar se perguntando como que a simulação da previsão de churn foi feito no Streamlit? Bom, uma vez que temos apenas 6 variáveis para o usuário escolher, torna-se necessário refazer o processo para o contexto dessas 6 variáveis. É importante ressaltar que essas variáveis foram escolhidas conforme os resultados de importância do XGBClassifier! Ah, é válido dizer que teremos um desempenho pior nesse modelo em comparação com o original por causa do menor volume de dados.")
    
    with st.echo():
        # Manter apenas as colunas relevantes
        columns_of_interest = ['Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Relationship_Count', 
                            'Months_Inactive_12_mon', 'Total_Trans_Amt', 'Contacts_Count_12_mon', 
                            'Attrition_Flag']
        bank_churners = bank_churners[columns_of_interest]
        bank_churners['Attrition_Flag'] = bank_churners['Attrition_Flag'].replace({'Attrited Customer': 1, 'Existing Customer': 0})

        # Separar em características e alvo
        X = bank_churners.drop('Attrition_Flag', axis=1)
        y = bank_churners['Attrition_Flag']

        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalizar os dados
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Treinar o modelo
        model = XGBClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)

        # Avaliar o modelo
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f'Acurácia: {accuracy:.2f}')
        with col2:
            st.write(f'Precisão: {precision:.2f}')
        with col3:
            st.write(f'Revocação: {recall:.2f}')
        with col4:
            st.write(f'F1-Score: {f1:.2f}')
    st.write("Conforme esperado, tivemos um desempenho pior do que o modelo que tinha todas as variáveis mas ainda está ok. É o suficiente para nós termos uma boa estimativa da previsão de churn na nossa ferramenta do Streamlit!.")

