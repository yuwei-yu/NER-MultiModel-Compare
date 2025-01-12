import torch.nn as nn
from transformers import BertModel, BertTokenizer
from utils import CRF
import torch


class MutilModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(MutilModel, self).__init__(*args, **kwargs)
        self.config = config
        if config['model'] == 'bert':
            self.model = BertModel.from_pretrained(config['pretrained_model'])
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrained_model'])
            self.fc = nn.Linear(config['hidden_size'], config['output_size'])
        if config['model'] == 'biLstm':
            self.tokenizer = nn.Embedding(config['vocab_size'], config['hidden_size'])
            self.model = nn.LSTM(config['input_size'], config['hidden_size'], config['num_layers'], bidirectional=True,
                                 batch_first=True, dropout=config['dropout_prob'])
            self.fc = nn.Linear(config['hidden_size'] * 2, config['output_size'])
        if config['CRF']:
            self.crf = CRF(num_tags=config['num_tags'], batch_first=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):

        input = self.tokenizer(input_ids)
        output, _ = self.model(input)
        output = self.fc(output)
        print(f"全连接层的输出：{output.shape}")
        if self.config["CRF"]:
            if labels is not None:
                mask = self.get_sequence_mask(input_ids)
                loss = self.crf.forward(output, labels, mask=mask)
                return -loss

            else:
                mask = self.get_sequence_mask(input_ids)
                pred = self.crf.decode(output, mask=mask)
                return pred
        else:
            if labels is not None:
                loss = self.loss(output.view(-1, self.config['num_tags']), labels.view(-1))
                return loss
            else:
                return output.argmax(dim=-1)

    def get_sequence_mask(self, labels):
        lens = (labels != 0).sum(-1)
        max_len = lens.max()
        mask = torch.arange(max_len).expand(len(lens), max_len).to(labels.device) < lens.unsqueeze(1)
