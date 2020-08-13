```python
import pandas as pd
import numpy as np
```

## Setup


```python
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

train_df.drop('Unnamed: 0', axis=1, inplace=True)

test_labels = test_df.columns.values.tolist()
tidy_train = train_df[test_labels]
```

## Cleaning the Data


```python
tidy_train['TP_SEXO'] = tidy_train['TP_SEXO'].map({'F':0, 'M':1}) 

tidy_train['TP_COR_RACA'] = tidy_train['TP_COR_RACA'].map({
    0:0, 6:0, 1:1, 2:0, 3:0, 4:1, 5:0
})
tidy_train['TP_ESCOLA'] = tidy_train['TP_ESCOLA'].map({
    1:0, 2:0, 3:1, 4:0
})

test_df['TP_SEXO'] = test_df['TP_SEXO'].map({'F':0, 'M':1}) 

test_df['TP_COR_RACA'] = test_df['TP_COR_RACA'].map({
    0:0, 6:0, 1:1, 2:0, 3:0, 4:1, 5:0
})

test_df['TP_ESCOLA'] = test_df['TP_ESCOLA'].map({1:0, 2:0, 3:1, 4:0})
```


```python
label = 'NU_NOTA_MT'

tidy_train[label] = train_df[label]

df = tidy_train.select_dtypes('number')
df.fillna(-1, inplace=True)

X = df.drop('NU_NOTA_MT', axis=1)
y = df['NU_NOTA_MT']
```

## Modeling


```python
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

## Selector will pick columns to use in the model
selector_model = Lasso(alpha=1.,normalize=True)
selector = SelectFromModel(selector_model, max_features=28, threshold=-np.inf)

Xtrain = df.drop('NU_NOTA_MT', axis=1)
ytrain = df['NU_NOTA_MT']

selector.fit(Xtrain, ytrain)

mask = selector.get_support()
print(f'Selected features:\n{Xtrain.columns[mask]}')
```

    Selected features:
    Index(['CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_SEXO', 'TP_COR_RACA',
           'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA',
           'TP_ENSINO', 'IN_TREINEIRO', 'TP_DEPENDENCIA_ADM_ESC', 'IN_BAIXA_VISAO',
           'IN_CEGUEIRA', 'IN_SURDEZ', 'IN_DISLEXIA', 'IN_DISCALCULIA',
           'IN_SABATISTA', 'IN_GESTANTE', 'IN_IDOSO', 'TP_PRESENCA_CN',
           'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH',
           'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1'],
          dtype='object')


## Prediction


```python
Xtest = test_df[X.columns]
Xtest.fillna(-1, inplace=True)

Xtrain2 = selector.transform(Xtrain)

model = RandomForestRegressor(criterion='mse', n_estimators=1000, max_depth=9, n_jobs=-1, random_state=0)
model.fit(Xtrain2, ytrain);

Xtest2 = selector.transform(Xtest)

ypred = model.predict(Xtest2)
```

## Post-processing


```python
answer = pd.DataFrame({'NU_INSCRICAO':test_df.NU_INSCRICAO,
                        'TP_PRESENCA_CN':test_df.TP_PRESENCA_CN,
                        'NU_NOTA_MT':ypred})

for index, row in answer.iterrows():
    if row['TP_PRESENCA_CN'] == 0:
        answer.loc[index,'NU_NOTA_MT'] = np.nan
answer['NU_NOTA_MT'] = answer['NU_NOTA_MT'].apply(lambda x: 0 if x < 250 else x)

answer.drop('TP_PRESENCA_CN', axis=1, inplace=True)
answer.to_csv('answer.csv', index=False)
```

## Submit


```python
! codenation submit -c enem-2
```

    [37mVersão: [32m1.0.14[0m
    
    [0m[36mUsando arquivo de configuração: /home/vcwild/.codenation.yml
    [0m[36m
    Executando testes...
    
    [0m{"score": 93.75923111050471}
    Preparando code review... 100% |████████████████████████████████████████|  [6s:0s][32m
    Códigos submetidos com sucesso! 
    [0m[32m
    Sua nota é: 93.759231
    [0m[36m
    Parabéns! Você superou este desafio!
    [0m[36m
    O que fazer agora?
    [0m[36m- Você pode continuar sua jornada escolhendo um novo desafio em https://www.codenation.com.br
    [0m[36m- Você pode ajudar outros desenvolvedores revisando códigos ou respondendo dúvidas no forum do desafio
    [0m[36m- Lembre-se que ensinar é uma ótima forma de ganhar reconhecimento e gera um bom karma ;)
    [0m
