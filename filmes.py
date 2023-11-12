import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
@st.cache_data(show_spinner=False)
def filmes_eda():
    # Definindo o título da página
    st.title("Análise Exploratória dos Dados - IMDB Top 250 Movies")
    # Escrevendo uma introdução à análise exploratória
    st.write("""
    Nesta seção, realizaremos uma análise exploratória dos dados contidos na lista dos 250 melhores filmes 
    segundo a avaliação do IMDB. Vamos explorar diversas características como ano de lançamento, avaliação, 
    gênero, entre outras, para entender melhor o que faz desses filmes os mais bem avaliados. Como uma pessoa apaixonada pela sétima arte, vai ser um prazer fazer esse estudo!
    """)
    df = pd.read_csv('IMDB Top 250 Movies.csv')
    df.rename(columns={
        'rank': 'Classificação',
        'name': 'Filme',
        'year': 'Ano',
        'rating': 'Avaliação',
        'genre': 'Gênero',
        'certificate': 'Classificação Indicativa',
        'run_time': 'Duração',
        'tagline': 'Slogan',
        'casts': 'Elenco',
        'directors': 'Diretores',
        'writers': 'Roteiristas'
    }, inplace=True)

    # Converter 'budget' e 'box_office' para valores numéricos (float), removendo qualquer caracter que não seja numérico.
    df.drop(['budget', 'box_office'], axis=1, inplace=True)
    df_index = df
    # Subtítulo antes de exibir a base de dados
    st.subheader("Base de Dados")
    df_index.set_index(['Classificação', 'Filme'], inplace=True)
    st.dataframe(df_index.head(250))
    st.write("""
    Caso queira baixar os dados para fazer suas próprias análises, basta fazer o download no canto superior direito da tabela. Antes de começarmos, quer saber se o seu filme favorito está nessa lista? pode ir na busca também localizada canto superior direito da tabela e pesquisar pelo seu filme. Sem mais delongas, vamos começar!
    """)
    # Antes de prosseguirmos, vamos limpar e converter os dados que são numéricos mas que podem estar em formatos inadequados, como 'run_time' e 'budget', 'box_office'.
    # Vamos excluir as colunas 'budget' e 'box_office' do DataFrame.
    def runtime_to_minutes(rt):
        try:
            if 'h' in rt:
                parts = rt.split('h')
                hours = int(parts[0].strip()) * 60
                minutes = int(parts[1].replace('m', '').strip()) if 'm' in parts[1] else 0
                return hours + minutes
            elif 'm' in rt:
                return int(rt.replace('m', '').strip())
            else:
                return None  # Retorna None para valores que não podem ser convertidos
        except ValueError:
            return None  # Retorna None para erros de conversão

    # Em seguida, você usaria essa função para limpar a coluna 'run_time' no seu DataFrame.
    df['Duração'] = df['Duração'].apply(runtime_to_minutes)

    # Subtítulo para a distribuição de frequência
    st.subheader("1. Qual é a década com mais filmes e com melhores avaliações entre o top 250 imdb?")
    # Para criar uma nova coluna 'Década' no DataFrame, podemos dividir o 'Ano' por 10, converter para int e multiplicar por 10 novamente.
    df['Década'] = (df['Ano'] // 10 * 10).astype(str) + 's'

    # Agrupando por ano para calcular a contagem e a média das avaliações
    yearly_data = df.groupby('Década').agg(Quantidade=('Década', 'size'), Avaliação_Média=('Avaliação', 'mean')).reset_index()

    # Inicialize o matplotlib figure e axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Crie o gráfico de barras para a quantidade de filmes por década no eixo primário
    sns.barplot(data=yearly_data, x='Década', y='Quantidade', ax=ax1, color='blue', label='Quantidade de Filmes')

    # Configura o eixo Y primário (ax1 para a quantidade de filmes)
    ax1.set_ylim(0, yearly_data['Quantidade'].max() + 35)  # Ajuste o espaço acima das barras se necessário

    # Crie um segundo eixo Y compartilhando o mesmo eixo X para a avaliação média
    ax2 = ax1.twinx()

    # Crie o gráfico de pontos para a avaliação média no eixo secundário
    sns.scatterplot(data=yearly_data, x='Década', y='Avaliação_Média', ax=ax2, color='red', s=100, label='Avaliação Média')

    # Adicione os valores de quantidade como texto acima das barras
    for index, value in enumerate(yearly_data['Quantidade']):
        ax1.text(index, value + 1, str(value), color='black', ha="center")

    # Configura o eixo Y secundário (ax2 para a avaliação média)
    ax2.set_ylim(yearly_data['Avaliação_Média'].min()-2.2, yearly_data['Avaliação_Média'].max()+0.9)  # Ajuste conforme necessário

    # Adicione os valores de avaliação média como texto acima dos pontos
    for index, value in enumerate(yearly_data['Avaliação_Média']):
        ax2.text(index, value + 0.05, f'{value:.2f}', color='black', ha="center", va="bottom")

    # Remova os ticks e labels dos eixos Y
    ax1.set_yticks([])
    ax2.set_yticks([])

    # Remova as descrições dos eixos Y
    ax1.set_ylabel('')
    ax2.set_ylabel('')

    # Adicione a legenda manualmente
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Remova a descrição do eixo X
    ax1.set_xlabel('')

    # Remova as linhas de grade
    ax1.grid(False)

    # Ajuste as margens para garantir que o layout esteja correto
    plt.tight_layout()

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

    st.write("""

    O gráfico apresenta a quantidade de filmes e as avaliações médias correspondentes por década. Podemos observar que **a década de 2000 foi a que teve o maior número de filmes lançados no Top250 IMDb**, com um total de 48 filmes. Seguida de perto, a década de 1990 e a de 2010 mostram também um número elevado de lançamentos, com 42 e 43 filmes, respectivamente.

    Quanto às avaliações médias, percebe-se uma estabilidade relativa ao longo das décadas, com ligeiras variações. **A década de 1990 apresentou a maior avaliação média de cerca de 8.40**, o que indica que, embora não tenha tido o maior número de filmes, os filmes dessa década foram muito bem recebidos pelo público e pela crítica. Realmente, um ótimo período para o cinema, vamos explorar melhor esta década no próximo tópico!

    Por fim, observa-se que a maioria dos filmes na lista do imdb 250 são recentes, estando posicionados depois do início da década de 1990.
    """)
    # Subtítulo
    st.subheader("2. Porque a década de 1990 foi tão marcante para o cinema? Vamos explorar!")
    st.markdown('<p style="font-size:20px;">Parte 2.1: Aprofundando cada ano da década de 1990:</p>', unsafe_allow_html=True)
        
    # Filter the dataset for the 1990s
    data_90s = df[(df['Ano'] >= 1990) & (df['Ano'] < 2000)]

    # Agrupar os dados por ano e calcular a média das avaliações e a contagem de filmes
    yearly_data_90s = data_90s.groupby('Ano').agg(Quantidade=('Ano', 'size'), Avaliação_Média=('Avaliação', 'mean')).reset_index()

    # Agrupar os dados por ano e calcular a média das avaliações e a contagem de filmes
    yearly_data_90s = data_90s.groupby('Ano').agg(Quantidade=('Ano', 'size'), Avaliação_Média=('Avaliação', 'mean')).reset_index()

    # Inicialize o matplotlib figure e axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Crie o gráfico de linhas para a quantidade de filmes por ano no eixo primário
    sns.lineplot(data=yearly_data_90s, x='Ano', y='Quantidade', ax=ax1, color='blue', marker='o', legend=False)

    # Configura o eixo Y primário (ax1 para a quantidade de filmes)
    ax1.set_ylim(0, yearly_data_90s['Quantidade'].max() + 25)  # Ajuste o espaço acima das linhas se necessário
    ax1.set_ylabel('Quantidade de Filmes', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Crie um segundo eixo Y compartilhando o mesmo eixo X para a avaliação média
    ax2 = ax1.twinx()

    # Crie o gráfico de linhas para a avaliação média por ano no eixo secundário
    sns.lineplot(data=yearly_data_90s, x='Ano', y='Avaliação_Média', ax=ax2, color='red', marker='o', legend=False)

    # Configura o eixo Y secundário (ax2 para a avaliação média)
    ax2.set_ylim(yearly_data_90s['Avaliação_Média'].min() - 0.5, yearly_data_90s['Avaliação_Média'].max() + 0.5)  # Ajuste conforme necessário
    ax2.set_ylabel('Avaliação Média', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Remover as linhas de grade e descrições dos eixos Y
    ax1.grid(False)
    ax1.set_yticks([])
    ax2.set_yticks([])

    for x, y in zip(yearly_data_90s['Ano'], yearly_data_90s['Quantidade']):
        ax1.text(x, y + 0.5, str(y), color='blue', ha='center', va='bottom')

    # Para a linha de avaliação média (vermelho)
    for x, y in zip(yearly_data_90s['Ano'], yearly_data_90s['Avaliação_Média']):
        ax2.text(x, y + 0.05, f"{y:.2f}", color='red', ha='center', va='bottom')


    # Defina o título e os rótulos do eixo X
    ax1.set_title('Avaliação Média e Quantidade de Filmes por Ano (1990s)')
    ax1.set_xlabel('Ano')

    # Ajuste o layout para garantir que tudo se encaixe
    plt.tight_layout()

    # Mostrar o gráfico no Streamlit
    st.pyplot(fig)

    st.write("""
    A análise do gráfico acima revela insights interessantes sobre os filmes do top 250 do IMDb da década de 1990. Nota-se, de início, que o ano de 1994 se destaca com a maior avaliação média, o que pode ser atribuído a filmes icônicos como **Shawshank Redemption**, **Forrest Gump**, **Pulp Fiction** e **The Lion King**. Esses filmes não apenas alcançaram altas avaliações por parte do público e críticos, mas também deixaram um legado duradouro na indústria cinematográfica.

    Por outro lado, 1995 foi marcante pelo maior número de filmes que entraram para o top 250, destacando-se o filme **Se7en**, um thriller psicológico com uma atmosfera sombria. Além disso, vale ressaltar o filme **Before Sunrise**, primeiro de uma das melhores trilogias do cinema, e o filme vencedor do oscar **Braveheart**.

    Essas observações enfatizam como os anos de 1994 e 1995 foram fundamentais para o cinema, com filmes que não só foram populares na época de seu lançamento, mas que continuam a ressoar com as audiências até hoje. É válido lembrar que existem outros filmes icônicos nesta década, sobretudo no famigerado ano de 1999!
    """)

    st.markdown('<p style="font-size:20px;">Parte 2.2: Vamos entender as quebras por gênero do filme e por classificação indicativa dos filmes da década de 1990:</p>', unsafe_allow_html=True)
    # Pré-processamento dos dados (supondo que o formato dos dados seja o mesmo que o carregado anteriormente)
    all_genres_series = data_90s['Gênero'].str.split(',', expand=True).stack()
    all_genre_counts = all_genres_series.value_counts()
    rating_counts = data_90s['Classificação Indicativa'].value_counts()

    # Função para criar gráficos
    def create_charts():
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Gráfico de Gêneros
        bars_genre = axes[0].bar(all_genre_counts.index, all_genre_counts.values, color='skyblue')
        axes[0].grid(False)
        axes[0].set_title('Quantidade de Filmes por Gênero (1990s)')
        axes[0].set_xlabel('Gênero')
        axes[0].set_yticks([])
        axes[0].tick_params(axis='x', rotation=45)
        for bar in bars_genre:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., 0.51*height, '%d' % int(height), ha='center', va='bottom')

        # Gráfico de Classificação Indicativa
        # Gráfico de pizza para Classificação Indicativa
        axes[1].pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'lightblue', 'lightgreen', 'gold', 'purple'])
        axes[1].set_title('Distribuição de Filmes por Classificação Indicativa (1990s)')

        plt.tight_layout()
        return fig

    # No Streamlit, usamos st.pyplot para mostrar a figura
    fig = create_charts()
    st.pyplot(fig)

    st.write("""
    Os gráficos fornecem uma visão esclarecedora sobre as tendências cinematográficas dos anos 90. O primeiro gráfico mostra que **Drama** é de longe o gênero mais comum, seguido de **Crime**. Esta predominância pode ser parcialmente explicada pela tendência dos filmes de drama e crime a abordarem temas mais complexos e maduros, que frequentemente resultam em classificações **'R'** para maiores de idade.

    Analisando o gráfico de distribuição de filmes por classificação indicativa, observamos que **66,7%** dos filmes são classificados como **'R'**. Este dado pode ser correlacionado com a prevalência de filmes de crime, que muitas vezes contêm conteúdo adulto, como violência e linguagem explícita, justificando esta classificação. 

    Embora o gênero de **Drama** lidere a contagem, sua natureza abrangente pode incluir subgêneros como **Crime** e **Suspense**, que muitas vezes são intensos e voltados para adultos, o que pode contribuir para o alto percentual da classificação **'R'**. Portanto, é possível concluir que a narrativa madura e as temáticas sérias são características que potencialmente influenciam tanto a classificação indicativa quanto a popularidade dos filmes.

    Por fim, é importante relembrar que essa análise é baseada em uma amostra de filmes altamente classificados e provavelmente não representa a totalidade da produção cinematográfica dos anos 90, visto que o natural é termos mais filmes com classificação indicativa mais leve, onde vários públicos são admitidos. Ainda assim, ela destaca uma tendência clara de que os filmes voltados para o público adulto com temáticas complexas e desafiadoras ganharam destaque durante essa década.
    """)

    st.markdown('<p style="font-size:20px;">Parte 2.3: Quais são os diretores e atores mais importantes desta década? </p>', unsafe_allow_html=True)
    # Expandir as colunas 'Diretores' e 'Elenco', que contêm listas de nomes, em linhas separadas
    directors_series = data_90s['Diretores'].str.split(',').explode()
    actors_series = data_90s['Elenco'].str.split(',').explode()

    # Obter os diretores e atores mais frequentes
    top_directors = directors_series.value_counts().head(5)
    top_actors = actors_series.value_counts().head(5)

    # Inicializar o matplotlib figure e axes para dois subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Gráfico de barras para diretores
    sns.barplot(y=top_directors.index, x=top_directors.values, ax=ax1, palette='Blues_d')
    ax1.set_title('Top 5 Diretores dos Anos 90 com Mais Filmes no Top250 IMDB')
    ax1.set_xlabel('Número de Filmes')
    ax1.set_ylabel('Diretores')

    # Gráfico de barras para atores
    sns.barplot(y=top_actors.index, x=top_actors.values, ax=ax2, palette='Reds_d')
    ax2.set_title('Top 5 Atores dos Anos 90 com Mais Filmes no Top250 IMDB')
    ax2.set_xlabel('Número de Filmes')
    ax2.set_ylabel('Atores')

    # Configure os eixos X para mostrar apenas números inteiros
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Ajuste do layout para evitar sobreposição e exibição clara dos gráficos
    plt.tight_layout()

    # Exibir os gráficos no Streamlit
    st.pyplot(fig)

    st.write("""
    Em relação aos **diretores**, a década de 1990 foi uma era dourada, marcada por uma onda de inovação narrativa e direção ousada. O gráfico da esquerda ilustra os **top 5 diretores dos anos 90** com mais filmes no Top250 do IMDb. **Steven Spielberg** lidera o ranking, um nome sinônimo de excelência cinematográfica, responsável por clássicos como **Jurassic Park**, um filme que representa a evolução da tecnologia e dos efeitos especiais, e **Schindler's List**, ambos lançados no mesmo ano e desafiando os limites do que era possível em termos de produção no cinema.

    **Frank Darabont** aparece com destaque, creditado por **The Shawshank Redemption**, frequentemente aclamado como um dos melhores filmes de todos os tempos e o filme com melhor avaliação no Top250 do IMDb.

    **Quentin Tarantino**, conhecido por seu estilo distintivo e diálogos marcantes, também figura na lista, com filmes como **Pulp Fiction**, que revolucionou a estrutura narrativa no cinema e influenciou uma geração de cineastas. Pessoalmente, meu diretor favorito!

    **David Fincher** e **Martin Scorsese**, ambos mestres em criar filmes intensos e psicologicamente complexos, contribuíram com obras primas como **Se7en** e **Goodfellas** respectivamente, filmes que continuam a ser estudados por sua abordagem inovadora e, particularmente, meus filmes favoritos destes diretores.

    A presença desses cineastas na lista de filmes mais bem avaliados dos anos 90 não é uma coincidência, mas uma confirmação de que o cinema da época era impulsionado por visionários que não tinham medo de explorar a condição humana e empurrar as fronteiras da arte cinematográfica. Eles não só moldaram a estética e as expectativas do cinema moderno, mas também criaram obras que permanecem relevantes até hoje.
    """)
    st.write("""
    Em relação aos **atores**, os anos 90 são frequentemente lembrados como uma época de ouro para performances icônicas no cinema, e o gráfico **Top 5 Atores dos Anos 90 com Mais Filmes no Top250 IMDb** reforça essa percepção. **Tom Hanks** lidera a lista, um ator cuja versatilidade e capacidade de capturar a humanidade de seus personagens fez com que ele se tornasse um dos favoritos para os espectadores. Com atuações inesquecíveis em filmes como **Forrest Gump** e **Saving Private Ryan**, Hanks não só ganhou o coração do público, mas também críticas altamente positivas e reconhecimento da Academia.

    **Kevin Spacey** segue de perto, com uma série de atuações que definiram sua carreira na década de 90, como em **American Beauty** e **The Usual Suspects**. Sua habilidade em retratar personagens complexos e muitas vezes moralmente ambíguos lhe rendeu um lugar de destaque entre os atores mais aclamados da época.

    **Morgan Freeman**, com sua presença poderosa e voz marcante, também figura nesta lista, trazendo profundidade a filmes como **Shawshank Redemption** e **Se7en**. A integridade e dignidade que Freeman infunde em seus personagens tornam suas atuações memoráveis e duradouras.

    **Steve Buscemi**, conhecido por seu estilo único e papéis ecléticos, marcou presença em obras de diretores proeminentes como Quentin Tarantino e os irmãos Coen, mostrando seu alcance como ator em clássicos como **Fargo** e **Reservoir Dogs**.

    Embora **Jack Angel** possa não ser um nome familiar para muitos, sua contribuição para o cinema dos anos 90 foi bastante significativa, emprestando sua voz a vários personagens animados em filmes icônicos como **Aladdin**, um clássico da Disney, e **Toy Story**, o primeiro filme do renomado estúdio Pixar.

    Esses atores exemplificam o talento extraordinário que emergiu e floresceu nos anos 90, cada um contribuindo para o legado do cinema com performances que permanecem influentes até hoje. Seus trabalhos não apenas dominaram as bilheterias, mas também ajudaram a definir o padrão de excelência na atuação para as gerações futuras.
    """)

    # Subtítulo 
    st.subheader("3. Qual é o gênero com mais filmes e com melhores avaliações entre o top 250 imdb?")

    df_genero = df
    df_genero['Gênero'] = df_genero['Gênero'].str.split(',')
    # Agora, vamos usar o método explode para criar uma nova linha para cada gênero
    df_genero = df_genero.explode('Gênero')

    # Agrupando por ano para calcular a contagem e a média das avaliações
    genero_data = df_genero.groupby('Gênero').agg(Quantidade=('Gênero', 'size'), Avaliação_Média=('Avaliação', 'mean')).reset_index()

    # Inicialize o matplotlib figure e axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Crie o gráfico de barras para a quantidade de filmes por década no eixo primário
    sns.barplot(data=genero_data, x='Gênero', y='Quantidade', ax=ax1, color='blue', label='Quantidade de Filmes')

    # Configura o eixo Y primário (ax1 para a quantidade de filmes)
    ax1.set_ylim(0, genero_data['Quantidade'].max() + 35)  # Ajuste o espaço acima das barras se necessário

    # Crie um segundo eixo Y compartilhando o mesmo eixo X para a avaliação média
    ax2 = ax1.twinx()

    # Crie o gráfico de pontos para a avaliação média no eixo secundário
    sns.scatterplot(data=genero_data, x='Gênero', y='Avaliação_Média', ax=ax2, color='red', s=100, label='Avaliação Média')

    # Adicione os valores de quantidade como texto acima das barras
    for index, value in enumerate(genero_data['Quantidade']):
        ax1.text(index, value + 1, str(value), color='black', ha="center")

    # Configura o eixo Y secundário (ax2 para a avaliação média)
    ax2.set_ylim(genero_data['Avaliação_Média'].min()-2.2, genero_data['Avaliação_Média'].max()+0.9)  # Ajuste conforme necessário

    # Adicione os valores de avaliação média como texto acima dos pontos
    for index, value in enumerate(genero_data['Avaliação_Média']):
        ax2.text(index, value + 0.05, f'{value:.2f}', color='black', ha="center", va="bottom")

    # Remova os ticks e labels dos eixos Y
    ax1.set_yticks([])
    ax2.set_yticks([])

    # Remova as descrições dos eixos Y
    ax1.set_ylabel('')
    ax2.set_ylabel('')

    # Adicione a legenda manualmente
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Remova a descrição do eixo X
    ax1.set_xlabel('')
    # Rotacione os rótulos do eixo X para melhor visualização
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Remova as linhas de grade
    ax1.grid(False)

    # Ajuste as margens para garantir que o layout esteja correto
    plt.tight_layout()

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)


    st.write("""
    Em uma primeira análise, é importante ressaltar que o gráfico apresenta a quantidade de filmes e as avaliações médias correspondentes por gênero. Fazendo uma análise mais detalhada do gráfico, temos:

    1. **Gênero com mais filmes**: O gênero **Drama** tem a maior quantidade de filmes no top 250, com um total de 177 filmes. Isso indica que filmes de drama são particularmente populares ou bem avaliados no IMDb. Além disso, conforme discutido anteriormente, a natureza abrangente deste gênero faz com que possa ser um subgênero bastante comum.

    2. **Avaliações médias**: Embora o gênero **Drama** tenha a maior quantidade de filmes, não possui a maior avaliação média. Gêneros como **Música**, **Crime** e **Sci-Fi** têm avaliações médias mais altas, embora representem uma quantidade menor de filmes no total.

    3. **Análise pontuais por gêneros específicos**:
    - **Crime** tem uma boa representação e avaliação média alta.
    - **Aventura** e **Ação** são dois gêneros que costumam andar juntos e ambos tem uma boa representação.
    - **Horror** tem uma avaliação média alta, mas com menos filmes no total quando comparado a outros gêneros.

    """)
    st.subheader("4. Qual é a relação dos gêneros dos filmes com a classificação indicativa?")
    # Agora, vamos criar um DataFrame de contagem cruzada para as classificações por gênero
    ct = pd.crosstab(df_genero['Gênero'], df_genero['Classificação Indicativa'])
    ct_reset = ct.reset_index()

    # Transforma o DataFrame de um formato largo para um formato longo
    ct_long = ct_reset.melt(id_vars='Gênero', var_name='Classificação', value_name='Quantidade')

    # Primeiro, vamos obter a lista de gêneros e classificações
    generos = ct_reset['Gênero'].unique()
    classificacoes = ct_reset.columns[1:]  # Ignora a primeira coluna que é 'Gênero'

    # Cria a figura do Matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))

    # Precisamos rastrear a 'base' para onde cada barra de um gênero começa no eixo Y
    base = np.zeros(len(generos))

    # Vamos iterar sobre as classificações para empilhar as barras
    for classificacao in classificacoes:
        # Aqui plotamos a barra para cada classificação
        ax.bar(
            generos, 
            ct_reset[classificacao],  # Altura da barra
            bottom=base,  # Inicio da barra no eixo Y
            label=classificacao
        )
        # Atualiza a 'base' adicionando a altura da barra atual, para que a próxima barra comece acima desta
        base += ct_reset[classificacao]

    # Adiciona títulos e rótulos
    ax.set_title('Gráfico de Barras Empilhadas por Gênero e Classificação')
    ax.set_xlabel('Gênero')
    ax.set_ylabel('Quantidade Acumulada')
    # Remova as linhas de grade
    ax.grid(False)
    ax.set_xticklabels(generos, rotation=45)
    ax.legend(title='Classificação Indicativa', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Streamlit - Exibe o gráfico no app
    st.pyplot(fig)

    st.write("""
    O gráfico oferece uma análise detalhada da relação entre os gêneros dos filmes e suas respectivas classificações indicativas. Há uma diversidade clara de classificações dentro de cada gênero, mas é possível notar a predominância da classificação indicativa **'R'** entre vários gêneros. 

    Por exemplo, o gênero **Drama** se destaca com a maior quantidade de filmes, e é notável a predominância da classificação **'R'**. Isso pode ser atribuído à tendência dos filmes de drama de explorarem temas adultos mais complexos e sérios que requerem uma abordagem madura, justificando assim a restrição de idade.

    Gêneros como **Ação** e **Aventura** apresentam uma variedade bem grande na  classificação indicativa, indicando uma mistura de filmes voltados tanto para o público jovem quanto para adultos.

    A presença de classificações como **'G'** e **'PG'** em gêneros como **Animação** e **Família** reflete a intenção desses filmes de serem acessíveis a todas as idades, frequentemente com conteúdo educativo ou de entretenimento leve.

    Em resumo, a análise do gráfico revela não apenas as preferências de conteúdo por gênero, mas também como as classificações indicativas podem influenciar a produção cinematográfica e a percepção do público em relação a cada gênero de filme. Será que existem outros fatores que afetam a percepção do público? Qual será a opinião do público em relação a duração de um filme? Vamos investigar!""")

    # Subtítulo 
    st.subheader("5. Como está a distribuição das durações dos filmes top 250 imdb?")
    # Crie o histograma para a coluna 'Duração'
    fig, ax = plt.subplots()
    sns.distplot(df['Duração'].dropna(), bins=30, ax=ax)

    # Configurações de título e rótulos
    ax.set_title('Distribuição da Duração dos Filmes')
    ax.set_xlabel('Duração (minutos)')
    ax.set_ylabel('Quantidade de Filmes')

    # Ajuste das marcações do eixo x para mostrar apenas números inteiros
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Remova as linhas de grade do eixo y e as marcações do eixo y
    ax.yaxis.grid(False)
    ax.set_yticks([])

    # Ajuste do layout para garantir que tudo se encaixe
    plt.tight_layout()

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

    st.write("""
    A distribuição da duração dos filmes, como mostrada no histograma, indica que a maioria dos filmes do Top 250 do IMDb têm uma duração concentrada entre **90 e 150 minutos**. Esta faixa é típica para longas-metragens e sugere uma preferência tanto de cineastas quanto de audiências por filmes que oferecem uma experiência completa dentro de um tempo de exibição razoável.

    Observamos também um pico em torno dos **120 minutos**, o que muitas pessoas consideram como a duração ideal para um filme conseguir contar uma história coesa com desenvolvimento de personagem e narrativa sem exigir um compromisso de tempo excessivo do espectador.

    Apesar da existência de filmes com durações menores que **90** minutos e maiores que **150** minutos, estes são menos comuns. Filmes com durações mais curtas podem não ter tempo suficiente para desenvolver enredos e personagens complexos, enquanto filmes mais longos podem ser desafiadores para manter a atenção da audiência e, por vezes, podem ser restritos a um público mais específico ou para experiências cinematográficas mais épicas. Mas será que existe alguma relação entre a duração do filme e a avaliação do público? Vamos descobrir!

    """)
    # Subtítulo 
    st.subheader("6. Existe alguma relação entre a duração do filme e a sua avaliação?")
    # Criação do scatterplot
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['Duração'], y=df['Avaliação'], ax=ax)

    # Título e rótulos do gráfico
    ax.set_title('Relação entre Duração e Avaliação dos Filmes')
    ax.set_xlabel('Duração (minutos)')
    ax.set_ylabel('Avaliação')

    # Removendo as linhas de grade para um estilo mais limpo, conforme solicitado
    ax.grid(False)

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

    st.write("""
    O gráfico acima exibe a relação entre a duração dos filmes e suas respectivas avaliações. O que se destaca de imediato é **que não há uma tendência clara que associe a duração dos filmes a avaliações mais altas ou mais baixas**. Os pontos estão distribuídos de maneira relativamente uniforme ao longo de uma ampla gama de durações, sem indicar que filmes mais longos ou mais curtos tendem a ser melhor ou pior avaliados.

    Para minha surpresa, os filmes que ultrapassam a duração de **150 minutos** - que muitas vezes são épicos cinematográficos ou filmes com narrativas mais complexas - não necessariamente recebem avaliações mais altas. De fato, embora esses filmes mais longos tenham mais tempo para desenvolvimento de personagens e da trama, há uma chance maior de cansaço por parte espectador.

    Por outro lado, filmes com duração inferior a **90 minutos** parecem ter avaliações abaixo da média, o que pode indicar que filmes extremamente curtos são menos propensos a serem incluídos entre os mais bem avaliados, devido a limitações na narrativa ou desenvolvimento de personagens. 

    """)

    # Subtítulo 
    st.subheader("7. Quais são os diretores com mais filmes e com melhor avaliação?")
    df_diretores = df
    df_diretores['Diretores'] = df_diretores['Diretores'].str.split(',')
    # Agora, vamos usar o método explode para criar uma nova linha para cada gênero
    df_diretores = df_diretores.explode('Diretores')

    # Agrupando por ano para calcular a contagem e a média das avaliações
    diretores_data = df_diretores.groupby('Diretores').agg(Quantidade=('Diretores', 'size'), Avaliação_Média=('Avaliação', 'mean')).reset_index()

    # Ordena os diretores primeiro pela 'Quantidade' e em seguida pela 'Avaliação_Média', ambos em ordem decrescente
    diretores_data = diretores_data.sort_values(by=['Quantidade', 'Avaliação_Média'], ascending=[False, False]).head(10)

    # Inicialize o matplotlib figure e axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Crie o gráfico de barras para a quantidade de filmes por década no eixo primário
    sns.barplot(data=diretores_data, x='Diretores', y='Quantidade', ax=ax1, color='blue', label='Quantidade de Filmes')

    # Configura o eixo Y primário (ax1 para a quantidade de filmes)
    ax1.set_ylim(0, diretores_data['Quantidade'].max() + 15)  # Ajuste o espaço acima das barras se necessário

    # Crie um segundo eixo Y compartilhando o mesmo eixo X para a avaliação média
    ax2 = ax1.twinx()

    # Crie o gráfico de pontos para a avaliação média no eixo secundário
    sns.scatterplot(data=diretores_data, x='Diretores', y='Avaliação_Média', ax=ax2, color='red', s=100, label='Avaliação Média')

    # Adicione os valores de quantidade como texto acima das barras
    for index, value in enumerate(diretores_data['Quantidade']):
        ax1.text(index, value + 1, str(value), color='black', ha="center")

    # Configura o eixo Y secundário (ax2 para a avaliação média)
    ax2.set_ylim(diretores_data['Avaliação_Média'].min()-2.2, diretores_data['Avaliação_Média'].max()+0.9)  # Ajuste conforme necessário

    # Adicione os valores de avaliação média como texto acima dos pontos
    for index, value in enumerate(diretores_data['Avaliação_Média']):
        ax2.text(index, value + 0.05, f'{value:.2f}', color='black', ha="center", va="bottom")

    # Remova os ticks e labels dos eixos Y
    ax1.set_yticks([])
    ax2.set_yticks([])

    # Remova as descrições dos eixos Y
    ax1.set_ylabel('')
    ax2.set_ylabel('')

    # Adicione a legenda manualmente
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Remova a descrição do eixo X
    ax1.set_xlabel('')
    # Rotacione os rótulos do eixo X para melhor visualização
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Remova as linhas de grade
    ax1.grid(False)

    # Ajuste as margens para garantir que o layout esteja correto
    plt.tight_layout()

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

    st.write("""
    O gráfico destaca os Top 10 diretores que não apenas tiveram vários filmes listados no Top 250 do IMDb, mas também alcançaram altas avaliações médias, refletindo a qualidade e a consistência de seu trabalho. Vamos explorar o legado deles e prometo dizer qual o meu filme favorito de cada um!

    1. **Christopher Nolan**: Com uma avaliação média impressionante, sendo a maior dentre esse Top 10. Nolan é conhecido por seus filmes complexos e inovadores, como **Inception** e **Interstellar**. Sua abordagem única para narrativas não lineares e temas filosóficos garante que seus filmes sejam não só populares, mas também altamente respeitados. É importante dizer que, durante a escrita deste texto, em 2023, Nolan dirigiu outra obra que entrou no Top 250 do IMDb: **Oppenheimer**. Meu filme favorito do diretor é **The Dark Knight**!.

    2. **Steven Spielberg**: Spielberg, o diretor com mais filmes na década de 90 na lista do IMDb, e um dos mais prolíficos e influentes, tem uma variedade diversificada de filmes aclamados, incluindo **E.T.** e **Indiana Jones and the Raiders of the Lost Ark**. Seu domínio em contar histórias e sua habilidade em capturar a imaginação do público são evidenciados por suas avaliações consistentemente altas. Meu filme favorito do diretor é **Jurassic Park**!

    3. **Stanley Kubrick**: Kubrick é lembrado por sua perfeição técnica e filmes que desafiam o público, como **2001: Uma Odisseia no Espaço** e **Eyes Wide Shut**. Sua avaliação média reflete o impacto duradouro de seu estilo visionário. Meu filme favorito do diretor é **A Clockwork Orange**!

    4. **Martin Scorsese**: Um mestre do cinema moderno, Scorsese é conhecido por seus dramas intensos e filmes de crime, como **Taxi Driver** e **Casino**. As altas avaliações de seus filmes atestam sua habilidade em explorar a complexidade da condição humana. Meu filme favorito do diretor é **Goodfellas**!

    5. **Akira Kurosawa**: Reverenciado por sua influência no cinema mundial, Kurosawa é um cineasta japonês que trouxe ao público obras-primas como **Rashomon** e **Seven Samurai**. Suas técnicas narrativas e estilísticas continuam a ser estudadas e admiradas por novas gerações de diretores. Meu filme favorito do diretor é **High and Low**!

    6. **Alfred Hitchcock**: Conhecido como o 'Mestre do Suspense', Hitchcock criou alguns dos thrillers mais icônicos, incluindo **Psycho** e **Vertigo**. As altas avaliações de seus filmes são um testemunho de seu talento em manter o público na beira do assento e aflitos. Meu filme favorito do diretor é **Rear Window**!

    7. **Quentin Tarantino**: Chegamos ao meu diretor favorito. Tarantino fez filmes como **Pulp Fiction** e **Kill Bill**, que são tanto homenagens quanto inovações nos gêneros que ele ama. Com diálogos memoráveis e cenas marcantes, o meu filme favorito do diretor é **Inglourious Basterds**!

    8. **Charles Chaplin**: Chaplin não só estrelou, mas também dirigiu algumas das comédias mudas mais queridas, como **Modern Times** e **The Kid**. Quando estivermos fazendo essa mesma análise para os atores, é bem provavél que ele apareça novamente, um verdadeiro gênio do cinema. Meu filme favorito do diretor é **Modern Times**!

    9. **Billy Wilder**: Um cineasta versátil, Wilder é responsável por clássicos como **Some Like It Hot** e **The Apartment**. Suas narrativas engenhosas e personagens bem desenvolvidos garantiram uma avaliação média elevada para seus filmes. Meu filme favorito do diretor é **The Apartment**!

    10. **Sergio Leone**: Por fim, mas não menos importante, temos Segio Leone, um cineasta italiano que foi pioneiro do gênero Western Spaghetti, ofereceu ao mundo filmes como **Once Upon a Time in the West**. Sua técnica estilizada e histórias épicas fizeram com que ele se tornasse a maior referência em filmes de faroeste. Meu filme favorito do diretor é **The Good, The Bad and The Ugly**!

    """)
    # Subtítulo 
    st.subheader("8. Quais são os atores com mais filmes e com melhor avaliação?")
    df_atores = df
    df_atores['Elenco'] = df_atores['Elenco'].str.split(',')
    # Agora, vamos usar o método explode para criar uma nova linha para cada gênero
    df_atores = df_atores.explode('Elenco')

    # Agrupando por ano para calcular a contagem e a média das avaliações
    atores_data = df_atores.groupby('Elenco').agg(Quantidade=('Elenco', 'size'), Avaliação_Média=('Avaliação', 'mean')).reset_index()

    # Ordena os diretores primeiro pela 'Quantidade' e em seguida pela 'Avaliação_Média', ambos em ordem decrescente
    atores_data = atores_data.sort_values(by=['Quantidade', 'Avaliação_Média'], ascending=[False, False]).head(10)

    # Inicialize o matplotlib figure e axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Crie o gráfico de barras para a quantidade de filmes por década no eixo primário
    sns.barplot(data=atores_data, x='Elenco', y='Quantidade', ax=ax1, color='blue', label='Quantidade de Filmes')

    # Configura o eixo Y primário (ax1 para a quantidade de filmes)
    ax1.set_ylim(0, atores_data['Quantidade'].max() + 15)  # Ajuste o espaço acima das barras se necessário

    # Crie um segundo eixo Y compartilhando o mesmo eixo X para a avaliação média
    ax2 = ax1.twinx()

    # Crie o gráfico de pontos para a avaliação média no eixo secundário
    sns.scatterplot(data=atores_data, x='Elenco', y='Avaliação_Média', ax=ax2, color='red', s=100, label='Avaliação Média')

    # Adicione os valores de quantidade como texto acima das barras
    for index, value in enumerate(atores_data['Quantidade']):
        ax1.text(index, value + 1, str(value), color='black', ha="center")

    # Configura o eixo Y secundário (ax2 para a avaliação média)
    ax2.set_ylim(atores_data['Avaliação_Média'].min()-2.2, atores_data['Avaliação_Média'].max()+0.9)  # Ajuste conforme necessário

    # Adicione os valores de avaliação média como texto acima dos pontos
    for index, value in enumerate(atores_data['Avaliação_Média']):
        ax2.text(index, value + 0.05, f'{value:.2f}', color='black', ha="center", va="bottom")

    # Remova os ticks e labels dos eixos Y
    ax1.set_yticks([])
    ax2.set_yticks([])

    # Remova as descrições dos eixos Y
    ax1.set_ylabel('')
    ax2.set_ylabel('')

    # Adicione a legenda manualmente
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Remova a descrição do eixo X
    ax1.set_xlabel('')
    # Rotacione os rótulos do eixo X para melhor visualização
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Remova as linhas de grade
    ax1.grid(False)

    # Ajuste as margens para garantir que o layout esteja correto
    plt.tight_layout()

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

    st.write("""
    Desta vez, vamos destacar o Top 10 atores que tiveram vários filmes listados no Top 250 do IMDb. Vamos fazer uma dinâmica parecida com a dos diretores:

    1. **Robert De Niro**: Com uma impressionante presença de 9 filmes no Top 250 e uma avaliação média de 8.38, De Niro se estabelece como uma lenda do cinema. Seus papéis variados e profundamente imersivos em filmes como **The Godfather Part II** e **Taxi Driver** demonstram sua versatilidade e dedicação à arte. Meu filme favorito do ator é **Goodfellas**!

    2. **Morgan Freeman**: Parace que o Morgan Freeman já é o nome mais citado neste texto hahaha, com 7 títulos no Top 250 e com uma média de 8.54, fica claro que suas performances em filmes como **The Shawshank Redemption** e **Se7en** deixaram uma marca indelével no cinema. Meu filme favorito do ator é **The Shawshank Redemption**!

    3. **Harrison Ford**: Conhecido por seus icônicos papéis em franquias como **Star Wars** e **Indiana Jones**, Ford também tem 7 filmes na lista, com uma avaliação média de 8.40 nestes filmes. Seu carisma e habilidade em capturar a essência de personagens heroicos são inegáveis. Meu filme favorito do ator é **The Empire Strikes Back**!

    4. **John Ratzenberger**: Embora menos conhecido em comparação com outros nomes, Ratzenberger tem uma forte presença em muitos filmes marcantes da Pixar, que frequentemente figuram no Top 250 do IMDb. Com uma média de 8.24, sua contribuição vocal para esses filmes amados é significativa. Meu filme favorito do ator é **The Incredibles**!

    5. **Michael Caine**: Com 6 filmes e uma média de 8.58, Cane é reconhecido por sua destreza em desempenhar uma variedade de papéis em filmes de diferentes gêneros em filmes como **Inception** e **Interstellar**. Meu filme favorito do ator é **The Dark Knight**!

    6. **Tom Hanks**: Outro ator com 6 filmes no Top 250, Hanks possui uma média de 8.45, refletindo filmes aclamados como **Forrest Gump** e **Saving Private Ryan**. Meu filme favorito do ator é **Toy Story**!

    7. **Christian Bale**: Conhecido por sua capacidade de transformação física e profundidade emocional, Bale possui uma média de 8.40 em seus 6 filmes no Top 250, incluindo **The Dark Knight** e **The Prestige**. Meu filme favorito do ator é **The Dark Knight**! 

    8. **Leonardo DiCaprio**: Com uma média de 8.37 e também com 6 filmes, DiCaprio tem mostrado um gerenciamento de carreira impressionante desde **Titanic** a **The Wolf of Wall Street**. Meu filme favorito do ator é **Django Unchained**!

    9. **Robert Duvall**: Conhecido por sua intensidade dramática e habilidade de mergulhar profundamente em seus personagens, Duvall tem 5 filmes no Top 250 e uma média de 8.62, o que faz com que ele seja o ator com maior média dentre esses 10 através de atuações formidáveis em **The Godfather** e **Apocalypse Now**. Meu filme favorito do ator é **The Godfather Part II**!

    10. **Alec Guinness**: Por fim, com uma média de 8.42 em seus 5 filmes, Guinness é eternizado por seu papel como Obi-Wan Kenobi em **Star Wars**, além de seus outros trabalhos notáveis. Meu filme favorito do ator é **The Empire Strikes Back**!

    Esses atores não só aparecem nos filmes mais bem avaliados, mas também contribuíram com performances que definiram suas carreiras e deixaram uma marca duradoura na história do cinema. Nota-se, além disso, que as descrições de filmes de alguns atores são bastante parecidas com a de alguns diretores que eu citei anteriormente. Veja, por exemplo, Robert De Niro e Martin Scorsese ou Michael Caine e Christopher Nolan, vamos dar uma olhada nessas parcerias icônicas do cinema?""")

    st.subheader("9. Quais são as principais parcerias entre atores/diretores e diretores/roteiristas ?")

    # Garanta que as colunas são strings e remova espaços em branco
    df['Elenco'] = df['Elenco'].astype(str).apply(lambda x: [i.strip() for i in x.split(',')])
    df['Diretores'] = df['Diretores'].astype(str).apply(lambda x: [i.strip() for i in x.split(',')])
    df['Roteiristas'] = df['Roteiristas'].astype(str).apply(lambda x: [i.strip() for i in x.split(',')])

    # Cria uma função para calcular as top 5 duplas
    def get_top_duplas(column1, column2):
        # Explode as colunas em linhas individuais
        temp_df = df.explode(column1).explode(column2)
        # Limpa os espaços em branco nas bordas das strings
        temp_df[column1] = temp_df[column1].str.strip()
        temp_df[column2] = temp_df[column2].str.strip()
        # Calcula as combinações e conta a frequência
        top_duplas = temp_df.groupby([column1, column2]).size().reset_index(name='Quantidade')
        # Ordena pelo número de filmes e pega o top 5
        top_duplas = top_duplas.sort_values(by='Quantidade', ascending=False).head(10)
        return top_duplas

    # Calcula as top 5 duplas para cada combinação e armazena em DataFrames separados
    top_elenco_diretor = get_top_duplas('Elenco', 'Diretores')
    top_diretor_roteirista = get_top_duplas('Diretores', 'Roteiristas')

    # Função para converter listas em strings e remover colchetes e aspas
    def clean_list_string(s):
        return s.replace("[", "").replace("]", "").replace("'", "")

    # Aplica a função de limpeza nos DataFrames
    top_elenco_diretor['Elenco'] = top_elenco_diretor['Elenco'].apply(clean_list_string)
    top_elenco_diretor['Diretores'] = top_elenco_diretor['Diretores'].apply(clean_list_string)
    top_elenco_diretor.rename(columns={
        'Elenco': 'Ator',
    }, inplace=True)
    top_diretor_roteirista['Diretores'] = top_diretor_roteirista['Diretores'].apply(clean_list_string)
    top_diretor_roteirista['Roteiristas'] = top_diretor_roteirista['Roteiristas'].apply(clean_list_string)

    # Utiliza st.beta_columns para exibir as tabelas lado a lado
    col1, col2 = st.columns(2)

    with col1:
    # Exibe as tabelas
        st.write("**Top 10 Parcerias entre Ator/Diretor**")
        st.dataframe(top_elenco_diretor,hide_index=True)
        st.write("""
    **Análise das Top 10 Parcerias entre Ator/Diretor no Top 250 de Filmes do IMDB**

    No topo da lista, **Michael Caine** e **Christopher Nolan** aparecem com **6 colaborações**, evidenciando uma parceria produtiva que inclui filmes aclamados como a trilogia **The Dark Knight** e **Inception**. 

    Notavelmente, **Charles Chaplin**, um pioneiro do cinema, figura tanto como ator quanto diretor, com **5 parcerias listadas com ele mesmo**, o que sublinha seu talento multifacetado e impacto duradouro no cinema.

    Outra parceria notória é a de **Christian Bale** e **Christopher Nolan**, com **4 filmes**. Juntos, eles criaram algumas das obras mais memoráveis do início do século 21, mostrando a versatilidade de Bale e a visão inovadora de Nolan.

    **Robert De Niro** e **Martin Scorsese** também marcam presença com **4 colaborações**, uma dupla que se tornou sinônimo de filmes de gângster de alta qualidade e dramas intensos, como **Goodfellas** e **Raging Bull**. Escrevendo esse texto em 2023, fico feliz em dizer que essa dupla acabou de lançar o filme **Killers of the Flower Moon**, completando 50 anos de parceria!

    A presença de **Akira Kurosawa** e seus atores frequentes, como **Toshiro Mifune**, **Takashi Shimura**, **Minoru Chiaki** e **Isao Kimura**, com **4, 4, 4 e 3 colaborações**, respectivamente, destaca a influência do cinema japonês e a habilidade de Kurosawa em extrair atuações poderosas de seu elenco recorrente.

    Por fim, **Morgan Freeman** aparece ao lado de **Christopher Nolan** com **3 filmes**, marcando outra colaboração significativa que contribuiu para o sucesso de filmes bem avaliados pelo público e crítica.

    """)

    with col2:
        st.write("**Top 10 Parcerias entre Diretor/Roteirista**")
        st.dataframe(top_diretor_roteirista,hide_index=True)
        st.write("""
    **Análise das Top 10 Parcerias entre Diretor/Roteirista no Top 250 de Filmes do IMDB**

    No topo, encontramos **Stanley Kubrick** e **Christopher Nolan**, ambos com **7 parcerias com eles mesmos**. Kubrick era conhecido por sua meticulosidade e genialidade, muitas vezes adaptando roteiros de obras existentes e infundindo-os com sua visão única. Nolan, por sua vez, frequentemente colabora com seu irmão, **Jonathan Nolan**, mas também assume a dupla função de diretor e roteirista, o que é evidenciado pelo mesmo número de colaborações tanto como roteirista quanto como diretor.

    **Akira Kurosawa**, uma lenda do cinema japonês, aparece duas vezes na lista: por suas próprias contribuições como roteirista e diretor, e por sua colaboração com **Hideo Oguni**, refletindo a importância de parcerias duradouras na criação de histórias envolventes.

    **Billy Wilder** e **Quentin Tarantino**, ambos com **5 colaborações**, são exemplos de diretores que frequentemente escrevem seus próprios roteiros, proporcionando uma visão coesa que se traduz em filmes que são tanto esteticamente distintos quanto narrativamente sólidos.

    E o que falar sobre **Charles Chaplin**? Não satisfeito em dirigir e atuar em seus filmes, ele também trabalha no roteiros destes. Um artista completo e uma das pessoas mais influentes da história do cinema.

    Por fim, **Hayao Miyazaki** do Studio Ghibli e **Pete Docter** da Pixar, cada um com **4 e 3 colaborações**, respectivamente, nos lembra que a animação também é um campo fértil para diretores/roteiristas deixarem uma marca inesquecível, criando mundos e histórias que ressoam em todas as idades.

    """)
    st.subheader("10. Considerações Finais")
    st.write("""
    Uau! Como uma pessoa apaixonada pelo cinema, aprendi muito fazendo a análise exploratória dos filmes listados no Top 250 do IMDb. Vamos para as considerações finais:

    1. **Décadas com mais filmes e com melhores avaliações entre o top250 IMDb**: Confesso que fiquei surpreso com a maioria dos filmes estando estando posicionados depois do início da década de 1990, visto que eu acreditava que existia uma grande quantidade de 'clássicos' figurando nesta lista, além disso é impressionante como os filmes dos anos 1990s e 2000s tem significância no Top250 IMDb.

    2. **Importância da década de 1990 para o cinema**: Uma década com vários filmes sensacionais! Com destaque para os anos de 1994 e 1995, tivemos muitos filmes de drama e crime abordando temas mais complexos e maduros. Os principais nomes desta época foram Steven Spielberg e Tom Hanks.

    3. **Gênero com mais filmes e com melhores avaliações entre o top250 IMDb**: O gênero drama foi disparado o que teve mais filmes. Particularmente, não foi tão surpreendente, devido ao fato de histórias mais aclamadas e com enredos complexos serem inerentes deste gênero.

    4. **Relação dos gêneros dos filmes com a classificação indicativa**: Para o contextos dos filmes Top250 IMDb, a classificação indicativa mais comum é a 'R', sobretudo para os gêneros de drama e crime.

    5. **Distribuição das durações dos filmes top250 IMDb**: Conforme eu estava esperando, a duração mais comum destes filmes está em torno dos 120 minutos, com algumas exceções como filmes com menos de 90 minutos e com mais de 150 minutos.

    6. **Relação entre a duração do filme e a sua avaliação**: Para minha surpresa, não há uma tendência clara que associe a duração dos filmes a avaliações mais altas ou mais baixas.

    7. **Diretores com mais filmes e com melhor avaliação**: Christopher Nolan foi o diretor com mais obras no Top250 IMDb e com uma avaliação média incrível de 8.56! 

    8. **Atores com mais filmes e com melhor avaliação**: Robert De Niro foi o ator com mais obras no Top250 IMDb, ao passo que Robert Duvall teve uma avaliação média notável de 8.62, sendo a maior entre os principais atores desta lista.

    9. **Principais parcerias entre atores/diretores e diretores/roteiristas**: A principal parceria entre ator e diretor é entre Michael Caine e Christopher Nolan, enquanto que é bastante comum o roteirista ser o próprio diretor do filme, além disso, claro, vale destacar a genialidade de Charles Chaplin como roteirista, diretor e ator.
    """)

    st.write('Caso queira compreender melhor como foram feitas cada análise ou discordar das minhas opiniões sobre os filmes (hahaha), sinta-se à vontade para entrar em contato por [Email](mailto:brunosouzabarros10@gmail.com) ou pelo [LinkedIn](https://www.linkedin.com/in/brunosouzabarros/).') 

