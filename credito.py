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
from simulacaochurn import simulacao_churn
from simulacaochurn import ferramenta_churn


def ferramenta_credito():
    min_values,max_values,model,X,scaler = simulacao_churn()
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
    
    ferramentas_churn = ferramenta_churn()