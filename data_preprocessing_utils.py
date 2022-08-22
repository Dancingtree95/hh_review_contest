import torch
import re

def rating_tokens(row):
  rating_columns = ['salary_rating', 'team_rating', 'managment_rating', 'career_rating', 'workplace_rating', 'rest_recovery_rating']
  res = '[A%d][B%d][C%d][D%d][E%d][F%d]' % tuple(row[rating_columns].tolist())
  return res
def make_text_features_with_rating_tokens(data, tokenizer):
  vac = data.position.values.tolist()
  vac = ['Должность: {}. Начало отзыва: '.format(v) for v in vac]
  pos = data.positive.values.tolist()
  pos = [v + p for v,p in zip(vac, pos)]
  neg = data.negative.values.tolist()
  ratings = data.apply(rating_tokens, axis=1)
  pos = [r + ' ' + p for r,p in zip(ratings, pos)]
  features = tokenizer(pos, neg, 
                    add_special_tokens=True, truncation=True, padding = 'max_length', max_length = 512, return_tensors = 'pt')
  return features['input_ids'], features['attention_mask'], features['token_type_ids']

def correct_zp(text):
    pattern = '\sз[^а-яА-ЯёЁ]{0,1}п[^а-яА-ЯёЁ]'
    matches = re.findall(pattern, text)
    if len(matches) == 0:
        return text
    for m in matches:
        try:
            text = re.sub(re.escape(m), ' зарплата' + m[-1], text)
        except:
            pass
    return text

def preprocess_data(df):
  df.positive = df.positive.fillna('')
  df.negative = df.negative.fillna('')
  df.position = df.position.fillna('Нет должности')
  df.positive = df.positive.apply(correct_zp)
  df.negative = df.negative.apply(correct_zp)
  return df




def collate_fn(batch):
  input_ids, attention_mask, token_type_ids,  labels = zip(*batch)
  features = {
      'input_ids': torch.stack(input_ids), 
      'attention_mask': torch.stack(attention_mask), 
      'token_type_ids' : torch.stack(token_type_ids)
  }
  labels = torch.stack(labels)
  return features, labels


def collate_fn_pred(batch):
  input_ids, attention_mask = zip(*batch)
  features = {
      'input_ids': torch.stack(input_ids), 
      'attention_mask': torch.stack(attention_mask), 
  }
  return features, None


