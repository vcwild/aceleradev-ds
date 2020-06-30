## Setup


```python
import pandas as pd 
```

## Ler dataset


```python
df = pd.read_csv('desafio1.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7000 entries, 0 to 6999
    Data columns (total 12 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   RowNumber                 7000 non-null   int64  
     1   id                        7000 non-null   object 
     2   sobrenome                 7000 non-null   object 
     3   pontuacao_credito         7000 non-null   int64  
     4   estado_residencia         7000 non-null   object 
     5   genero                    7000 non-null   object 
     6   idade                     7000 non-null   int64  
     7   nivel_estabilidade        7000 non-null   int64  
     8   saldo_conta               7000 non-null   float64
     9   numero_produtos           7000 non-null   int64  
     10  possui_cartao_de_credito  7000 non-null   int64  
     11  membro_ativo              7000 non-null   int64  
    dtypes: float64(1), int64(7), object(4)
    memory usage: 656.4+ KB


## Obter medidas de dispersão


```python
dff = df.groupby('estado_residencia')['pontuacao_credito']
aux = pd.DataFrame({})
aux[0] = dff.agg(lambda x: x.value_counts().index[0]) #mode
aux[1] = dff.median()
aux[2] = dff.mean()
aux[3] = dff.std()
aux.index.name = None
aux = aux.transpose()
aux = aux.rename(index = {
    0: 'moda',
    1: 'mediana',
    2: 'media',
    3: 'desvio_padrao'
})

aux
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PR</th>
      <th>RS</th>
      <th>SC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>moda</th>
      <td>850.000000</td>
      <td>850.000000</td>
      <td>850.000000</td>
    </tr>
    <tr>
      <th>mediana</th>
      <td>650.000000</td>
      <td>650.000000</td>
      <td>653.000000</td>
    </tr>
    <tr>
      <th>media</th>
      <td>648.961294</td>
      <td>651.105143</td>
      <td>649.537653</td>
    </tr>
    <tr>
      <th>desvio_padrao</th>
      <td>98.607186</td>
      <td>95.136598</td>
      <td>97.233493</td>
    </tr>
  </tbody>
</table>
</div>



## Exportar para JSON


```python
out = aux[['SC', 'RS', 'PR']]
out.to_json('submission.json')
```
