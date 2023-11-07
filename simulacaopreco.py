import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import io
from sklearn.linear_model import LassoCV
from scipy.stats import skew, norm
from sklearn.model_selection import cross_val_score

def simulacao_preco():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
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

    # Função para conversão de pés quadrados para metros quadrados
    def sqft_to_sqm(sqft):
        return sqft * 0.092903

    # Função para conversão de metros quadrados para pés quadrados
    def sqm_to_sqft(sqm):
        return sqm / 0.092903

    with st.form("prediction_form"):
    # Coletando as entradas do usuário
        GrLivArea_sqm = st.slider('Área habitável acima do solo (em metros quadrados):', min_value=int(sqft_to_sqm(300)), max_value=int(sqft_to_sqm(6000)))
        LotArea_sqm = st.slider('Tamanho do lote (em metros quadrados):', min_value=int(sqft_to_sqm(1000)), max_value=int(sqft_to_sqm(12000)))

        MSZoning = st.selectbox('Classificação de Área:', ['Agricultura', 'Comercial', 'Residencial em Vilas Flutuantes', 'Industrial', 
                                                                'Residencial de Alta Densidade', 'Residencial de Baixa Densidade', 
                                                                'Parque Residencial de Baixa Densidade', 'Residencial de Média Densidade'])

        Neighborhood = st.selectbox('Bairro:', [
            'Bloomington Heights', 'Bluestem', 'Briardale', 'Brookside',
            'Clear Creek', 'College Creek', 'Crawford', 'Edwards', 'Gilbert',
            'Iowa DOT and Rail Road', 'Meadow Village', 'Mitchell', 'North Ames',
            'Northridge', 'Northpark Villa', 'Northridge Heights', 'Northwest Ames',
            'Old Town', 'South & West of Iowa State University',
            'Sawyer', 'Sawyer West', 'Somerset', 'Stone Brook', 'Timberland', 'Veenker'
        ])


        OverallQual = st.selectbox('Qualidade geral do imóvel:', ['Muito Excelente', 'Excelente', 'Muito Bom', 'Bom', 
                                                    'Acima da Média', 'Média', 'Abaixo da Média', 'Pouco Fraco', 'Fraco', 'Muito Fraco'])
        submit_button = st.form_submit_button("Prever Preço")

    if submit_button:
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

        neighborhood_dict = {
            'Bloomington Heights': 'Blmngtn',
            'Bluestem': 'Blueste',
            'Briardale': 'BrDale',
            'Brookside': 'BrkSide',
            'Clear Creek': 'ClearCr',
            'College Creek': 'CollgCr',
            'Crawford': 'Crawfor',
            'Edwards': 'Edwards',
            'Gilbert': 'Gilbert',
            'Iowa DOT and Rail Road': 'IDOTRR',
            'Meadow Village': 'MeadowV',
            'Mitchell': 'Mitchel',
            'North Ames': 'Names',
            'Northridge': 'NoRidge',
            'Northpark Villa': 'NPkVill',
            'Northridge Heights': 'NridgHt',
            'Northwest Ames': 'NWAmes',
            'Old Town': 'OldTown',
            'South & West of Iowa State University': 'SWISU',
            'Sawyer': 'Sawyer',
            'Sawyer West': 'SawyerW',
            'Somerset': 'Somerst',
            'Stone Brook': 'StoneBr',
            'Timberland': 'Timber',
            'Veenker': 'Veenker'
        }

        OverallQual_dict = {
            'Muito Excelente': 10,
            'Excelente': 9,
            'Muito Bom':8,
            'Bom':7,
            'Acima da Média':6,
            'Média':5,
            'Abaixo da Média':4,
            'Pouco Fraco':3,
            'Fraco':2,
            'Muito Fraco':1
        }

        MSZoning = MSZoning_dict[MSZoning]
        Neighborhood = neighborhood_dict[Neighborhood]
        OverallQual = OverallQual_dict[OverallQual]
        # Convertendo de volta para pés quadrados antes de criar o dataframe
        GrLivArea = sqm_to_sqft(GrLivArea_sqm)
        LotArea = sqm_to_sqft(LotArea_sqm)
        # Verificação se a área habitável é maior que o tamanho do lote
        if GrLivArea_sqm > LotArea_sqm:
            st.error("A área habitável tem que ser menor que o tamanho do lote!")
        else:
            # Criar o dataframe de entrada
            input_data = pd.DataFrame({
                'GrLivArea': [GrLivArea],
                'LotArea': [LotArea],
                'MSZoning': [MSZoning],
                'Neighborhood': [Neighborhood],
                'OverallQual': [OverallQual]
            })

            # Aplicar a transformação de log nas variáveis numéricas com assimetria maior que 0.75
            input_data[['GrLivArea', 'LotArea']] = np.log1p(input_data[['GrLivArea', 'LotArea']])

            # Criando one-hot encoding para os dados de entrada
            input_data_encoded = pd.get_dummies(input_data)

            # Criando um dataframe com todas as colunas necessárias preenchidas com 0
            input_data_final = input_data_encoded.reindex(columns=final_train.columns.drop('SalePrice'), fill_value=0)


            # Se necessário, adicione colunas faltantes com valores 0
            for col in final_train.columns:
                if col not in input_data_final.columns and col != 'SalePrice':
                    input_data_final[col] = 0

            # Certifique-se de que a ordem das colunas está correta
            input_data_final = input_data_final[final_train.columns.drop('SalePrice')]

            # Assegure-se de excluir a coluna 'SalePrice' se ela estiver presente
            if 'SalePrice' in input_data_final.columns:
                input_data_final.drop('SalePrice', axis=1, inplace=True)

            predicted_price = np.expm1(model_lasso_5variaves.predict(input_data_final))
            st.write(f'Previsão de Preço: ${predicted_price[0]:,.2f}')