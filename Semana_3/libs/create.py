# pylint: disable=E1120
import streamlit as st
import pandas as pd 
import altair as alt 


def hist(column, df):
  chart = alt.Chart(df, width = 600).mark_bar().encode(
    alt.X(column, bin = True),
    y = 'count()', tooltip = [column, 'count()']
  ).interactive()

  return chart


def bars(num_col, cat_col, df):
  bars = alt.Chart(df, width = 600).mark_bar().encode(
    x = alt.X(num_col, stack = 'zero'),
    y = alt.Y(cat_col),
    tooltip = [cat_col, num_col]
  ).interactive()

  return bars


def boxplot(num_col, cat_col, df):
  boxplot = alt.Chart(df, width = 600).mark_boxplot().encode(
    x = num_col,
    y = cat_col
  )

  return boxplot


def scatterplot(x, y, color, df):
  scatterplot = alt.Chart(df, width = 800, height = 400).mark_circle().encode(
    alt.X(x),
    alt.Y(y),
    color = color,
    tooltip = [x, y]
  ).interactive()

  return scatterplot


def corrplot(df, num_cols):
  corr_data = (df[num_cols]).corr().stack().reset_index().rename(
    columns = {
      0: 'correlation', 
      'level_0': 'variable', 
      'level_1': 'variable2'
    }
  )
  corr_data['correlation_label'] = corr_data['correlation'].map('{:.2f}'.format)
  base = alt.Chart(corr_data, width = 500, height = 500)\
    .encode(x = 'variable2:O', y = 'variable:O')
  text = base.mark_text().encode(
    text = 'correlation_label', 
    color = alt.condition(
      alt.datum.correlation > 0.5, 
      alt.value('white'), 
      alt.value('black')
    )
  )
  # The correlation heatmap itself
  corr_plot = base.mark_rect().encode(color = 'correlation:Q')

  return corr_plot + text