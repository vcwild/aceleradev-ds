
# Desafio 1

Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.

Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.

> Obs.: Por favor, não modifique o nome das funções de resposta.

## _Set up_ da análise


```python
import pandas as pd
import numpy as np
```


```python
bf = pd.read_csv("black_friday.csv")
```

## Inicie sua análise a partir daqui

## Questão 1

Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple (n_observacoes, n_colunas).


```python
def q1():

    return bf.shape
```


```python
q1()
```




    (537577, 12)



## Questão 2

Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.


```python

def q2():
    fit = bf[["User_ID", "Gender", "Age"]]
    females = fit["Gender"] == 'F'
    young = fit["Age"] == '26-35'
    fit = fit[females & young]
    
    return int(fit["User_ID"].nunique())
```


```python
q2()
```




    545



## Questão 3

Quantos usuários únicos há no dataset? Responda como um único escalar.


```python
def q3():

    return int(bf["User_ID"].nunique())
```


```python
q3()
```




    5891



## Questão 4

Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.


```python
def q4():

    return int(bf.dtypes.nunique())
```


```python
q4()
```




    3



## Questão 5

Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.


```python
def q5():
    rows = bf.shape[0]
    complete_rows = bf.dropna(axis='index').shape[0]
    na_rows = rows - complete_rows
    
    return float(na_rows / rows)
```


```python
q5()
```




    0.6944102891306734



## Questão 6

Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.


```python
def q6():

    return int(bf.isna().sum().max())
```


```python
q6()
```




    373299



## Questão 7

Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.


```python
def q7():

    return int(bf["Product_Category_3"].mode())
```


```python
q7()
```




    16



## Questão 8

Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.


```python
def q8():
    pur = bf.Purchase
    normal = (pur - pur.min()) / (pur.max() - pur.min())
    
    return float(normal.mean())
```


```python
q8()
```




    0.3847939036269795



## Questão 9

Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.


```python
def q9():
    pur = bf.Purchase
    z = (pur - pur.mean()) / pur.std()

    return int(((z >= -1) & (z <= 1)).dropna().sum())
```


```python
q9()
```




    348631



## Questão 10

Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).


```python
def q10():
    products = bf[["Product_Category_2", "Product_Category_3"]]
    null_products = products[products["Product_Category_2"].isna()]
    
    return bool(null_products["Product_Category_2"].equals(null_products["Product_Category_3"]))
```


```python
q10()
```




    True


