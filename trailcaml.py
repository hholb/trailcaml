from collections import namedtuple
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchmetrics import Accuracy
import lightning as L


ImageSize = namedtuple("ImageSize", ["x", "y"])


class TrailCaML(L.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        loss_fn=nn.BCEWithLogitsLoss(),
        fine_tune_after=5,
        lr_reduction=1e2,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters("lr", "fine_tune_after", "lr_reduction")
        self.loss_fn = loss_fn

        self.backbone = resnet18(weights="DEFAULT")
        
        # Freeze all backbone layers initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Adapt input layer for grayscale, preserving pretrained weights
        old_weight = self.backbone.conv1.weight.data
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1.weight.data = old_weight.sum(dim=1, keepdim=True)

        # Replace final layer
        num_filters = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_filters, 4)

        self.accuracy_metrics = nn.ModuleDict({
            'train_acc': Accuracy(task="multilabel", num_labels=4),
            'val_acc': Accuracy(task="multilabel", num_labels=4),
            'test_acc': Accuracy(task="multilabel", num_labels=4)
        })


    def unfreeze_backbone(self, from_layer=6):
        ct = 0
        for child in self.backbone.children():
            if ct >= from_layer:
                for param in child.parameters():
                    param.requires_grad = True
            ct += 1

    def forward(self, x):
        return self.backbone(x)

    def on_train_epoch_start(self):
        # Start fine-tuning after specified number of epochs
        if self.current_epoch == self.hparams.fine_tune_after:
            self.unfreeze_backbone()
            # Adjust learning rate for fine-tuning
            for param_group in self.trainer.optimizers[0].param_groups:
                self.hparams.lr = self.hparams.lr / self.hparams.lr_reduction
                param_group['lr'] = self.hparams.lr


    def _shared_step(self, batch, batch_idx, stage):
        X, y = batch
        logits = self(X)
        loss = self.loss_fn(logits, y)

        # Compute binary predictions for accuracy metric
        with torch.no_grad():
            preds = torch.sigmoid(logits) > 0.5

        self.accuracy_metrics[f"{stage}_acc"](preds, y.bool())

        self.log(
            f"{stage}_loss", 
            loss, 
            prog_bar=(stage == 'train'),
            on_epoch=True,
        )
        self.log(
            f"{stage}_accuracy",
            self.accuracy_metrics[f"{stage}_acc"],
            prog_bar=True,
            on_epoch=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer =torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=2,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def on_train_epoch_end(self):
        self.accuracy_metrics['train_acc'].reset()

    def on_validation_epoch_end(self):
        self.accuracy_metrics['val_acc'].reset()

    def on_test_epoch_end(self):
        self.accuracy_metrics['test_acc'].reset()
