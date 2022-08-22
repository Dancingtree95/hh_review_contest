import copy
import torch
from transformers import BertModel


def deleteEncodingLayers(model, num_layers_to_keep): 
    oldModuleList = model.encoder.layer
    newModuleList = torch.nn.ModuleList()

    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.layer = newModuleList

    return copyOfModel


class BertMeanMaxPooling(torch.nn.Module):
  def __init__(self, from_pretrained, cls_dim, n_classes, h_dim = None, p = 0.5, n_layers = 10):
    super(BertMeanMaxPooling, self).__init__()
    self.bert = BertModel.from_pretrained(from_pretrained)
    self.bert = deleteEncodingLayers(self.bert, n_layers)
    if h_dim:
      self.classifier = torch.nn.Sequential(torch.nn.Dropout(p), torch.nn.Linear(cls_dim*2, h_dim), torch.nn.ReLU(), torch.nn.Dropout(p), torch.nn.Linear(h_dim, n_classes))
    else:
      self.classifier = torch.nn.Sequential(torch.nn.Dropout(p), torch.nn.Linear(cls_dim*2, n_classes))
    
  
  def forward(self, input_ids, attention_mask, token_type_ids):
    outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids)
    last_hidden_state = outputs.last_hidden_state
    last_hidden_state_cls = last_hidden_state[:,0,:] #ошибка: cls токен содержится в pooler_output, а не в last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state[:,1:,:] * input_mask_expanded[:,1:,:], 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    last_hidden_state[input_mask_expanded == 0] = -1e9
    max_embedings = torch.max(last_hidden_state[:,1:,:], 1)[0]
    embed = torch.cat((mean_embeddings, max_embedings), -1)
    logit = self.classifier(embed)
    return logit