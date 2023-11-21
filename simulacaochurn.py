import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    # Avaliar o modelo
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Streamlit UI
    st.subheader('Ferramenta de Previsão de Churn:')

    with st.form("prediction_form"):
        Total_Trans_Ct = st.slider('Quantas Transações o Cliente fez nos Últimos 12 meses?',
                                int(min_values['Total_Trans_Ct']), int(max_values['Total_Trans_Ct']))
        Total_Trans_Amt = st.slider('Qual é o Montante Transacionado pelo Cliente nos Últimos 12 meses?',int(min_values['Total_Trans_Amt']),int(max_values['Total_Trans_Amt']))
        Total_Revolving_Bal = st.slider('Qual é o Saldo Rotativo no Cartão de Crédito do Cliente?',int(min_values['Total_Revolving_Bal']),int(max_values['Total_Revolving_Bal']))
        Total_Relationship_Count = st.slider('Qual é o Número Total de Produtos que o Cliente tem com o Banco?',int(min_values['Total_Relationship_Count']),int(max_values['Total_Relationship_Count']))
        Months_Inactive_12_mon = st.slider('Quantos Meses em que o Cliente Esteve Inativo nos Últimos 12 meses?',int(min_values['Months_Inactive_12_mon']),int(max_values['Months_Inactive_12_mon']))
        Contacts_Count_12_mon = st.slider('Quantas Vezes o Cliente Entrou em Contato com o Banco nos Últimos 12 meses?',int(min_values['Contacts_Count_12_mon']),int(max_values['Contacts_Count_12_mon']))


        # Botão de previsão
        if st.form_submit_button('Prever Churn'):
            # Preparar dados para previsão
            input_data = pd.DataFrame([[Total_Trans_Ct, Total_Revolving_Bal, Total_Relationship_Count, 
                                        Months_Inactive_12_mon, Total_Trans_Amt, Contacts_Count_12_mon]],
                                    columns=X.columns)
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)
            prediction_proba = model.predict_proba(input_data_scaled)[:, 1]*100
            
            # Mostrar resultado e probabilidade
            if prediction == 0:
                st.success(f'O cliente provavelmente não vai dar Churn. Probabilidade de churn: {prediction_proba[0]:.0f}%')
            else:
                st.error(f'O cliente provavelmente vai dar Churn. Probabilidade de churn: {prediction_proba[0]:.0f}%')

    st.write("Quer descobrir como foi feita essa ferramenta? Me acompanhe nos próximos passos!")
    # Mostrar métricas de desempenho




