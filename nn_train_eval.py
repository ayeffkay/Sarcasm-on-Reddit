from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchtext.data import BucketIterator
from transformers.optimization import AdamW
import copy
from sklearn import metrics
import matplotlib.pyplot as plt


class ClassificationModel(LightningModule):
    def __init__(self, model, train_data, valid_data, test_data, lr=1e-3, batch_size=128, is_bert=False):
        super().__init__()
        self.model = model
        self.is_bert = is_bert

        self.auc = pl.metrics.functional.classification.auroc
        self.roc = pl.metrics.functional.classification.roc

        self.criterion = nn.CrossEntropyLoss()
        self.roc_curve = None

        self.lr = lr
        self.batch_size = batch_size

        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data

    
    def train_dataloader(self):
        return BucketIterator(self.train_data, batch_size=self.batch_size, 
                              shuffle=True, 
                              sort_key=lambda x: len(x.comment))

    
    def val_dataloader(self):
        return BucketIterator(self.valid_data, batch_size=self.batch_size, 
                       shuffle=False, 
                       sort_key=lambda x: len(x.comment))
    
    def test_dataloader(self):
        return BucketIterator(self.test_data, batch_size=self.batch_size, 
                              shuffle=False, 
                              sort_key=lambda x: len(x.comment))


    def configure_optimizers(self):
        if self.is_bert:
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {'optimizer': optimizer, 
                'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min'), 
                'monitor': 'valid_loss', 
                'interval': 'batch', 
                'frequency': 1}


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        logits = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return logits


    @staticmethod
    def class_prob(logits):
        return torch.softmax(logits, dim=1)

    
    @staticmethod
    def class_id(probs):
        return torch.argmax(probs, dim=1)



    def make_step(self, batch, mode):
        with torch.set_grad_enabled(mode=='train'):
            logits = self.forward(input_ids=batch.comment, 
                                  attention_mask=vars(batch).get('attention_mask'),
                                  token_type_ids=vars(batch).get('token_type_ids'))
            loss = self.criterion(logits, batch.label)

            probs = self.class_prob(logits)
            classes = self.class_id(probs)
            labels = batch.label

            batch_dict = {'loss': loss, 'probs': probs, 
                          'classes': classes, 'labels': labels}

            return batch_dict

    
    def compute_epoch_metrics(self, outputs, mode):
        with torch.no_grad():
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            true = torch.cat([x['labels'] for x in outputs])
            labels = copy.deepcopy(true.cpu().numpy())
            probs = torch.cat([x['probs'] for x in outputs], dim=0)
            classes = torch.cat([x['classes'] for x in outputs]).cpu().numpy()


            epoch_dict = {mode + '_loss': avg_loss.item(), 
                          mode + '_accuracy': metrics.accuracy_score(labels, classes), 
                          mode + '_precision': metrics.precision_score(labels, classes), 
                          mode + '_recall': metrics.recall_score(labels, classes), 
                          mode + '_f1': metrics.f1_score(labels, classes), 
                          mode + '_auc': self.auc(probs[:, 1], true).item()
                        }

            for name, value in epoch_dict.items():
                self.log(name, value, prog_bar=True, logger=False)
                self.logger.experiment.log_metric(name, x=self.current_epoch, y=value)
                
            if mode == 'test':
                epoch_dict.update({'probs': probs, 'labels': true})

            return epoch_dict


    def training_step(self, batch, batch_idx):
        batch_dict = self.make_step(batch, mode='train')
        return batch_dict


    def training_epoch_end(self, outputs):
        epoch_dict = self.compute_epoch_metrics(outputs, 'train')


    def validation_step(self, batch, batch_idx):
        batch_dict = self.make_step(batch, mode='valid')
        return batch_dict


    def validation_epoch_end(self, outputs):
        epoch_dict = self.compute_epoch_metrics(outputs, 'valid')
        
    
    def test_step(self, batch, batch_idx):
        batch_dict = self.make_step(batch, mode='test')
        return batch_dict

    
    def plot_roc(self, auc):
        fpr, tpr, thresholds = list(map(lambda x: x.cpu().numpy(), self.roc_curve))
        fig = plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC for class 1 (area = {:.2f})'.format(auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC on test')
        plt.legend(loc="lower right")

        self.logger.experiment.log_image('ROC-AUC on test', fig)
        plt.close(fig)

    
    @staticmethod
    def gen_text_output(outputs):
        head = []
        values = []
        for log, output in sorted(outputs.items()):
            if log == 'probs' or log == 'labels':
                continue
            head.append(log)
            values.append('{:.2f}'.format(output))
        s_head = ','.join(head)
        s_values = ','.join(values)
        return '\n'.join([s_head, s_values])


    def test_epoch_end(self, outputs):
        epoch_dict = self.compute_epoch_metrics(outputs, 'test')
        self.roc_curve = self.roc(epoch_dict['probs'][:, 1], epoch_dict['labels'])
        self.plot_roc(epoch_dict['test_auc'])
        str_output = self.gen_text_output(epoch_dict)
        self.logger.experiment.log_text('test_outputs', str_output)
