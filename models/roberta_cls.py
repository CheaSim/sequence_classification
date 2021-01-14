import torch
import torch.nn as nn
import pytorch_lightning as pl


class ModelForClassification(pl.LightningModule):
    def __init__(self, roberta, ):
        super(ModelForClassification, self).__init__()
        self.roberta = roberta
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids):
        logits, _  = self.roberta(input_ids)

        return logits

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):            
        logits = self.forward(batch['input_ids'])
        loss = self.loss_fn(logits, batch['label']).mean()

        return {'loss': loss, 'log' : {'train_loss': loss}}


    def validation_step(self, batch, batch_idx):            
        logits = self.forward(batch['input_ids'])
        loss = self.loss_fn(logits, batch['label']).mean()
        acc = (logits.argmax(-1) == batch['label']).float()

        return {'loss': loss, 'log' : {'val_acc': acc}}
    


