import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm

def postprocess(pred_proba):
  pred_proba = np.array(pred_proba)
  pred = np.zeros_like(pred_proba, dtype = 'int')
  for i, row in enumerate(pred_proba):
    mask = row >= 0.5
    if mask.sum() > 0:
      pred[i, mask] = 1
    else:
      idx = np.argmax(row)
      pred[i, idx] = 1
  return pred


def get_predict(model, data, binary, return_labels = False):
  global DEVICE
  model.eval()
  sigmoid = torch.nn.Sigmoid()
  pred = []
  if return_labels:
    truth = []
  for features, labels in data:
    for unit in features.keys():
        features[unit] = features[unit].to(DEVICE)
    with torch.no_grad():
      logit = model(**features)
    proba = sigmoid(logit).cpu().tolist()
    pred.extend(proba)
    if return_labels:
      truth.extend(labels.tolist())
  if not binary:
    pred = postprocess(pred).tolist()
  else:
    pred = torch.where(torch.tensor(pred) > 0.5,1, 0 )
  if return_labels:
    truth = np.array(truth)
    truth = np.where(truth > 0.5, 1, 0)
    return pred, truth
  else:
    return pred


def get_optimizer_grouped_parameters(model, bert_lr, task_lr):
    no_decay = ["bias", "LayerNorm.weight"]
    lr = bert_lr['bert_lr']
    embed_lr = bert_lr['embed_lr']
    num_layers = bert_lr['n_layers']
    lr_decay = bert_lr['lr_decay']
    n_reinit = bert_lr['n_reinit']
    weight_decay = 0.01
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "bert" not in n],
            "weight_decay": 0.0,
            "lr": task_lr,
        },
    ]
    optimizer_grouped_parameters += [
        {
            'params': [p for n,p in model.named_parameters() if 'embeddings' in n], 
            'lr':embed_lr, 
            'weight_decay':0.0
        }
    ]
    # initialize lrs for every layer
    layers = list(model.bert.encoder.layer)
    layers.reverse()
    for i, layer in enumerate(layers):
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": task_lr if i < n_reinit else lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": task_lr if i < n_reinit else lr,
            },
        ]
        lr *= lr_decay
    
    return optimizer_grouped_parameters


def model_eval(model, data, binary = False):
  model.eval()
  pred, truth = get_predict(model, data, binary, return_labels = True)
  pred = np.array(pred)
  truth = np.where(np.array(truth) > 0.5, 1, 0)
  if not binary:
    return f1_score(truth, pred, average = 'samples')
  else:
    return f1_score(truth, pred)


def train_loop(model, train_data, train_config, eval_function = None, eval_data = None):
  bert_lr = train_config['bert_lr']
  task_lr = train_config['task_lr']
  epoch = train_config['epoch']
  
  opt_param = train_config['opt_param_fn'](model, bert_lr, task_lr)
  eval_scores = [0]
  total_steps = epoch * len(train_data)
  loss_fn = train_config['loss_fn']
  optimizer = AdamW(opt_param, eps = 1e-8, correct_bias=True)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=900, num_training_steps=total_steps)
  val_score = 0
  if eval_function:
    val_score = eval_function(model, eval_data)
    eval_scores.append(val_score)
  bar = tqdm(total = total_steps)
  exp_loss = 0
  for ep in range(epoch):
    step = 0
    for features, labels in train_data:
      model.train()
      for unit in features.keys():
        features[unit] = features[unit].to(DEVICE)
      #labels = transform_y(labels)
      labels = labels.to(DEVICE)
      model.zero_grad()
      logits = model(**features)
      loss = loss_fn(logits.squeeze(), labels)
      if exp_loss == 0:
        exp_loss = loss.item()
      else:
        exp_loss = 0.95 * exp_loss + 0.05 * loss.item()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
      optimizer.step()
      scheduler.step()
      step += 1
      if step == len(train_data) // 2:
        val_score = eval_function(model, eval_data)
        eval_scores.append(val_score)
      if step % 10 == 0 and step is not 0:
        bar.update(10)
        bar.set_description('Loss: {:.3f}/{:.3f}'.format(exp_loss, val_score))
    if eval_function:
      val_score = eval_function(model, eval_data)
      eval_scores.append(val_score)
    bar.set_description('Loss: {:.3f}/{:.3f}'.format(exp_loss, val_score))
  return eval_scores

