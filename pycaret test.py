from pycaret.classification import *
import pandas as pd

df = pd.read_csv('./NMLoanDefault.csv')

df.head()

exp_clf101 = setup(data = df, target = 'TARGET', session_id=42) 
compare_models()

rf = create_model('rf')
print(rf)

et = create_model('et')
print(et)

catboost = create_model('catboost')
print(catboost)

lightgbm = create_model('lightgbm')
print(lightgbm)