import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class SimpleNN(nn.Module):
    def __init__(self, input_size, dropout_prob=0.3, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, input_size), 
                                 nn.LayerNorm(input_size),
                                 nn.GELU(), 
                                 nn.Dropout(dropout_prob),
                                 nn.Linear(input_size, input_size // 4),
                                 nn.LayerNorm(input_size // 4),
                                 nn.GELU(),
                                 nn.Dropout(dropout_prob),
                                 nn.Linear(input_size // 4, n_classes)
                                 )
        
    def forward(self, input_weights, **kwargs):
        logits = self.net(input_weights)
        return logits
    

class SimpleRNN(nn.Module):
    def __init__(self, rnn_class, num_embeddings=10000, input_size=128, hidden_size=64, num_layers=1, 
                 bidirectional=False, dropout_prob=0.3, use_hidden=True, 
                 bias=True, batch_first=True, n_classes=2, from_pretrained=False, vectors=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_directions = int(bidirectional) + 1
        self.hidden_size = hidden_size
        self.use_hidden = use_hidden

        if from_pretrained:
            num_embeddings, input_size = vectors.shape
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(vectors))
        else:
            self.embedding = nn.Embedding(num_embeddings, input_size)

        self.rnn = rnn_class(input_size, hidden_size, num_layers, bias, batch_first, dropout_prob, bidirectional)
        self.drop = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, n_classes)
    
    def forward(self, input_ids, **kwargs):
        b_size, seq_len = input_ids.shape
        emb = self.embedding(input_ids)
        output, h_n = self.rnn(emb)

        if isinstance(self.rnn, torch.nn.modules.rnn.LSTM):
            h_n = h_n[0]
        h_n = h_n.view(self.num_layers, self.num_directions, b_size, self.hidden_size).permute(2, 0, 1, 3)
        output = output.view(b_size, seq_len, self.num_directions, self.hidden_size)

        if self.use_hidden:
            if self.num_directions == 2:
                h_n = torch.mean(h_n, dim=2)
            clf_input = h_n[:, -1, :].squeeze(dim=1)
        else:
            if self.num_directions == 2:
                output = torch.mean(output, dim=2)
            clf_input = output[:, -1, :].squeeze(dim=1)

        pooled_output = self.drop(clf_input)
        logits = self.fc(pooled_output)
        return logits
    


class SimpleCNN(nn.Module):
    def __init__(self, num_embeddings=10000, input_size=128, n_classes=2, from_pretrained=False, vectors=None, **kwargs):
        super().__init__()

        if from_pretrained:
            num_embeddings, input_size = vectors.shape
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(vectors))
        else:
            self.embedding = nn.Embedding(num_embeddings, input_size)

        self.convnet = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=input_size // 2, kernel_size=3, padding=1), 
                                     nn.ReLU(),
                                     nn.MaxPool1d(kernel_size=3, padding=1),
                                     nn.Flatten())
        self.fc = nn.Sequential(nn.AdaptiveMaxPool1d(512),
                                nn.LayerNorm(512),
                                nn.GELU(), 
                                nn.Linear(512, 128), 
                                nn.GELU(), 
                                nn.Linear(128, n_classes))
        
    def forward(self, input_ids, **kwargs):
        emb = self.embedding(input_ids).permute(0, 2, 1)
        conv_out = self.convnet(emb).unsqueeze(1)
        logits = self.fc(conv_out).squeeze(1)

        return logits
    



class CombinedCNN(nn.Module):
    def __init__(self, num_embeddings=10000, input_size=64, kernels=[3], dropout_prob=0.3, n_classes=2, from_pretrained=False, vectors=None, **kwargs):
        super().__init__()

        self.kernels = kernels
        if from_pretrained:
            num_embeddings, input_size = vectors.shape
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(vectors))
        else:
            self.embedding = nn.Embedding(num_embeddings, input_size)

        self.conv_blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=input_size, 
                                                                  out_channels=input_size // 2, 
                                                                  kernel_size=kernel), 
                                                        nn.BatchNorm1d(input_size // 2),
                                                        nn.ReLU(), 
                                                        nn.MaxPool1d(kernel)) for kernel in kernels])
        self.fc = nn.Sequential(nn.AdaptiveMaxPool1d(1024), 
                                nn.LayerNorm(1024),
                                nn.GELU(),
                                nn.Dropout(dropout_prob),
                                nn.Linear(1024, 128),
                                nn.GELU(),
                                nn.Linear(128, n_classes)
                                )
    
    def forward(self, input_ids, **kwargs):
        b_size, seq_len = input_ids.shape
        emb = self.embedding(input_ids).permute(0, 2, 1)
        conv_out = torch.cat([conv_block(emb) for conv_block in self.conv_blocks], dim=2).view(b_size, 1, -1)
        logits = self.fc(conv_out).squeeze(1)
        return logits
    
    


class RCNN(nn.Module):
    def __init__(self, rnn_class, num_embeddings=10000, input_size=128, hidden_size=64, num_layers=1, 
                 dropout_prob=0.3, bias=True, batch_first=True, kernel=3, n_classes=2, from_pretrained=False, vectors=None):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 2

        if from_pretrained:
            num_embeddings, input_size = vectors.shape
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(vectors))
        else:
            self.embedding = nn.Embedding(num_embeddings, input_size)

        self.rnn = rnn_class(input_size, hidden_size, num_layers, bias, batch_first, dropout_prob, bidirectional=True)
        self.drop = nn.Dropout(dropout_prob)
        
        self.cnn1 = self.get_cnn_block(hidden_size, hidden_size, kernel)
        self.cnn2 = self.get_cnn_block(hidden_size, hidden_size, kernel)

        self.fc = nn.Sequential(nn.Tanh(),
                                nn.Dropout(dropout_prob), 
                                nn.Linear(self.hidden_size, n_classes))


    @staticmethod
    def get_cnn_block(input_size, output_size, kernel):
        return nn.Sequential(nn.Conv1d(input_size, output_size, kernel), 
                             nn.GELU(), 
                             nn.MaxPool1d(kernel))

    def forward(self, input_ids, **kwargs):
        b_size, seq_len = input_ids.shape

        emb = self.embedding(input_ids)
        rnn_out, h_n = self.rnn(emb)
        rnn_out = rnn_out.view(b_size, seq_len, self.num_directions, self.hidden_size)

        rnn_out1 = rnn_out[:, :, 0, :].permute(0, 2, 1)
        rnn_out2 = rnn_out[:, :, 1, :].permute(0, 2, 1)


        conv_out1 = self.cnn1(rnn_out1).unsqueeze(3)
        conv_out2 = self.cnn2(rnn_out2).unsqueeze(3)

        mean_conv = torch.cat([conv_out1, conv_out2], dim=3).mean(dim=(2, 3))
        logits = self.fc(mean_conv)


        return logits
    


class BertForSeqClf(nn.Module):
    def __init__(self, pretrained_model_name, freeze_head=False, num_classes=2):
        super().__init__()

        config = BertConfig.from_pretrained(pretrained_model_name)
        config.return_dict = True
        config.num_labels = num_classes
        self.config = config
        
        self.body = BertModel.from_pretrained(pretrained_model_name, config=config, add_pooling_layer=False)

        self.head = nn.Linear(config.hidden_size, config.num_labels)
        self.head.apply(self.init_weights)

        if freeze_head:
            self.head.apply(self.freeze)


    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()


    @staticmethod
    def freeze(module):
        for param in module.parameters():
            param.requires_grad = False


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        enc_out = self.body(input_ids, attention_mask, token_type_ids)
        cls_token = enc_out.last_hidden_state[:, 0]
        logits = self.head(cls_token)

        return logits
