# pylint: disable=E1120
import libs
import streamlit as st
import pandas as pd 
import altair as alt 


def main():
  st.image('logo.png', width = 200)
  st.title('AceleraDev Data Science')
  st.subheader('Semana 3 - Análise Exploratória de Dados')
  st.image('https://cdn.pixabay.com/photo/2019/02/05/07/52/pixel-cells-3976295_960_720.png', width = 200)

  file = st.file_uploader('Escolha a base de dados que deseja analisar (.csv)', type = 'csv')

  if file is not None:
    st.subheader('Estatística descritiva univariada')
    df = pd.read_csv(file)
    aux = pd.DataFrame({'cols': df.columns, 'types': df.dtypes})
    num_cols = list(aux[aux['types'] != 'object']['cols'])
    cols_object = list(aux[aux['types'] == 'object']['cols'])
    cols = list(df.columns)
    col = st.selectbox('Selecione a coluna:', num_cols)

    if col is not None:
      st.markdown('Selecione o que deseja analisar:')
      
      mean = st.checkbox('Média')
      if mean:
        st.markdown(df[col].mean())

      median = st.checkbox('Mediana')
      if median:
        st.markdown(df[col].median())
      
      std = st.checkbox('Desvio Padrão')
      if std:
        st.markdown(df[col].std())
      
      kurt = st.checkbox('Kurtosis')
      if kurt:
        st.markdown(df[col].kurtosis())
      
      skew = st.checkbox('Skewness')
      if skew:
        st.markdown(df[col].skew())
      
      describe = st.checkbox('Describe')
      if describe:
        st.table(df[num_cols].describe().transpose())
    
    st.subheader('Visualização dos dados')
    st.image('https://cdn.pixabay.com/photo/2016/12/22/13/35/analytics-1925495_960_720.png', width = 200)
    st.markdown('Selecione a visualização')
    
    hist = st.checkbox('Histograma')
    if hist:
      col_num = st.selectbox('Selecione a Coluna Numérica: ', num_cols, key = 'unique')
      st.markdown('Histograma da coluna: ' + str(col_num))
      st.write(libs.hist(col_num, df))

    barras = st.checkbox('Gráfico de barras')
    if barras:
      col_num_bars = st.selectbox('Selecione a coluna numerica: ', num_cols, key = 'unique')
      col_cat_bars = st.selectbox('Selecione uma coluna categorica : ', cols_object, key = 'unique')
      st.markdown('Gráfico de barras da coluna ' + str(col_cat_bars) + ' pela coluna ' + col_num_bars)
      st.write(libs.bars(col_num_bars, col_cat_bars, df))

    boxplot = st.checkbox('Boxplot')
    if boxplot:
      col_num_box = st.selectbox('Selecione a Coluna Numérica: ', num_cols, key = 'unique')
      col_cat_box = st.selectbox('Selecione uma coluna categórica: ', cols_object, key = 'unique')
      st.markdown('Boxplot ' + str(col_cat_box) + ' pela coluna ' + col_num_box)
      st.write(libs.boxplot(col_num_box, col_cat_box, df))
    
    scatterplot = st.checkbox('Scatterplot')
    if scatterplot:
      col_num_x = st.selectbox('Selecione o valor de x ', num_cols, key = 'unique')
      col_num_y = st.selectbox('Selecione o valor de y ', num_cols, key = 'unique')
      col_color = st.selectbox('Selecione a coluna para a cor', cols)
      st.markdown('Selecione os valores de x e y')
      st.write(libs.scatterplot(col_num_x, col_num_y, col_color, df))
    
    corr = st.checkbox('Correlação')
    if corr:
      st.markdown('Gráfico de correlação das colunas numéricas')
      st.write(libs.corrplot(df, num_cols))

  
if __name__ == '__main__':
  main()
    
    
