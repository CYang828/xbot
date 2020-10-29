import torch
from torch import nn
from transformers import BertModel
from torch.nn.parameter import Parameter

class JointBERT(nn.Module):
    def __init__(self, model_config, device, intent_dim, intent_weight=None):
        super(JointBERT, self).__init__()
        self.intent_num_labels = intent_dim
        self.device = device
        self.intent_weight = torch.tensor(intent_weight) if intent_weight is not None else torch.tensor([1.]*intent_dim)

        print(model_config['pretrained_weights'])
        self.bert = BertModel.from_pretrained(model_config['pretrained_weights'])
        self.dropout = nn.Dropout(model_config['dropout'])
        self.context = model_config['context']
        self.finetune = model_config['finetune']
        self.context_grad = model_config['context_grad']
        self.hidden_units = model_config['hidden_units']
        if self.hidden_units > 0:
            if self.context:
                self.intent_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
            else:
                self.intent_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
            nn.init.xavier_uniform_(self.intent_hidden.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear( self.bert.config.hidden_size, self.intent_num_labels)
            else:
                self.intent_classifier = nn.Linear(self.bert.config.hidden_size, self.intent_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)

        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)

    def forward(self, word_seq_tensor, word_mask_tensor, intent_tensor=None):
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():   ##torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:##更新参数
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        pooled_output = outputs[1]


        if self.hidden_units > 0:
            pooled_output = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)

        # return intent_logits, intent_loss   #训练的时候使用这行代码
        return intent_logits  #预测的时候的返回值

