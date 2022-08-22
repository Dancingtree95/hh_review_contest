import numpy as np
import pandas as pd

def save_submit(pred):
  global mlb
  global test
  pred = np.array(pred)
  res = [','.join([str(k) for k in s]) for s in mlb.inverse_transform(pred)]
  submit = pd.DataFrame(columns = ['review_id', 'target'])
  submit['review_id'] = test['review_id']
  submit['target'] = res
  submit.to_csv('submit.csv')


