import streamlit as st
from ferramentapreco import ferramenta_preco
from filmes import filmes_eda

def main():
    # Configuração da página
    st.set_page_config(page_title="Portfólio Bruno Barros", layout="wide")
    pages = {
        "Sobre mim": page_resumo_profissional,
        "Ferramenta de previsão dos preços de casas": page_ferramenta_previsao,
        "Análise exploratória de dados (EDA) de filmes":page_ferramenta_filme
    }
    st.sidebar.image('Dados_Imagem.png')
    page = st.sidebar.radio("Escolha uma página:", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page]()

def page_resumo_profissional():

    # Cabeçalho
    st.title('Portfólio de Bruno Souza Barros')

    # Introdução
    st.write("""
        Bem vindo ao meu portfólio de Ciência de Dados! Sou apaixonado por enfrentar e solucionar problemas complexos e como um entusiasta Data Driven, entendo que os dados são a chave para decisões corretas e resultados impactantes. Este site é uma vitrine das minhas habilidades de programação, estatística e negócios aplicadas a projetos práticos.
    """)

    # Objetivos do Site
    st.header('Objetivo do portfólio')
    st.write("""
        O objetivo deste portfólio é apresentar meu trabalho em Ciência de Dados, destacando projetos significativos, 
        minha abordagem analítica e como eu estou aplicando aprendizados obtidos em dados na esfera profissional, acadêmica e pessoal.
    """)

    # Projetos de Destaque
    st.header('Projetos')
    # Colunas para imagem e botão
 
    st.subheader('1. Ferramenta de previsão dos preços de casas')
    st.write('Esse projeto tem como objetivo prever preços de casas utilizando modelos de regressão. Basta acessar a página **Ferramenta de previsão dos preços de casas** para visualizar o projeto completo.')
    st.subheader('2. Análise exploratória de dados de filmes')
    st.write('Esse projeto tem como objetivo fazer uma EDA do Top250 filmes do IMDb. Basta acessar a página **Análise exploratória de dados (EDA) de filmes** para visualizar o projeto completo.')
    
    # Contato/Call to Action
    st.header('Entre em Contato')
    st.write('Caso queira discutir oportunidades ou colaborações, sinta-se à vontade para entrar em contato por [Email](mailto:brunosouzabarros10@gmail.com) ou pelo [LinkedIn](https://www.linkedin.com/in/brunosouzabarros/).') 

    # Incluir elementos visuais como gráficos ou ícones (opcional)
    # st.image('caminho_para_sua_imagem.png', caption='Ciência de Dados em Ação')


def page_ferramenta_previsao():
    ferramenta_precos = ferramenta_preco()

def page_ferramenta_filme():
    filme_eda = filmes_eda()

if __name__ == "__main__":
    main()
