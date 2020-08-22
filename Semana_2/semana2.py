# pylint: disable=E1120
import streamlit as st
import pandas as pd
import base64

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

def main():
    st.image('logo.png', width= 200)
    st.title('AceleraDev Data Science')
    st.subheader('Semana 2 - Pré-processamento de Dados em Python')
    st.image('https://media.giphy.com/media/KyBX9ektgXWve/giphy.gif', width=200)
    file = st.file_uploader('Escolha a base de dados que deseja analisar (.csv)', type = 'csv')

    if file is not None:
        st.subheader('Analisando os dados')
        df = pd.read_csv(file)
        st.markdown('**Número de linhas:**')
        st.markdown(df.shape[0])
        st.markdown('**Número de colunas:**')
        st.markdown(df.shape[1])
        st.markdown('**Visualizando o dataframe**')
        number = st.slider('Escolha o numero de colunas que deseja ver', min_value=1, max_value=20)
        st.dataframe(df.head(number))
        st.markdown('**Nome das colunas:**')
        st.markdown(list(df.columns))
        filtered = pd.DataFrame({
            'names': df.columns, 
            'types': df.dtypes, 
            'NA #': df.isna().sum(), 
            'NA %': (df.isna().sum() / df.shape[0]) * 100
        })
        st.markdown('**Contagem dos tipos de dados:**')
        st.write(filtered.types.value_counts())
        st.markdown('**Nomes das colunas do tipo int64:**')
        st.markdown(list(filtered[filtered['types'] == 'int64']['names']))
        st.markdown('**Nomes das colunas do tipo float64:**')
        st.markdown(list(filtered[filtered['types'] == 'float64']['names']))
        st.markdown('**Nomes das colunas do tipo object:**')
        st.markdown(list(filtered[filtered['types'] == 'object']['names']))
        st.markdown('**Tabela com coluna e percentual de dados faltantes :**')
        st.table(filtered[filtered['NA #'] != 0][['types', 'NA %']])
        
        st.subheader('Inputaçao de dados númericos:')
        percent = st.slider('Escolha o limite de percentual faltante limite para as colunas vocë deseja inputar os dados', min_value=0, max_value=100)
        list_cols = list(filtered[filtered['NA %']  < percent]['names'])
        select_method = st.radio('Escolha um metodo abaixo :', ('Média', 'Mediana'))
        st.markdown(f'Você selecionou: {select_method}')

        if select_method == 'Média':
            df_inputed = df[list_cols].fillna(df[list_cols].mean())
        if select_method == 'Mediana':
            df_inputed = df[list_cols].fillna(df[list_cols].median())
        filtered_inputed = pd.DataFrame({
            'names': df_inputed.columns, 
            'types': df_inputed.dtypes, 
            'NA #': df_inputed.isna().sum(),
            'NA %': (df_inputed.isna().sum() / df_inputed.shape[0]) * 100
        })
        st.table(filtered_inputed[filtered_inputed['types'] != 'object']['NA %'])
        st.subheader('Dados Inputados, faça download abaixo: ')
        st.markdown(get_table_download_link(df_inputed), unsafe_allow_html=True)


if __name__ == '__main__':
	main()
