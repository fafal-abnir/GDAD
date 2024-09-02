import torch
import hydra.utils
from dataclasses import dataclass
from torchmetrics.classification import BinaryAveragePrecision
from torch.nn import BCEWithLogitsLoss

import pytorch_lightning as L


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    metric: str = "torchmetrics.classification.BinaryAveragePrecision"
    loss_fn: str = "torch.nn.BCEWithLogitsLoss"


class LightningGNN(L.LightningModule):
    def __init__(self, model, config: TrainingConfig):
        super().__init__()
        self.model = model
        self.learning_rate = config.learning_rate
        self.metric = BinaryAveragePrecision()
        self.loss_fn = BCEWithLogitsLoss()
        self.save_hyperparameters()
        self.automatic_optimization = False

    def reset_loss(self, loss):
        self.loss_fn = loss()

    # def forward(self, x, edge_index, edge_label_index, previous_embeddings=None, num_current_edges=None,
    #             num_previous_edges=None):
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_label_index = data.edge_label_index
        previous_embeddings = data.previous_embeddings
        num_current_edges = data.num_current_edges
        num_previous_edges = data.num_previous_edges
        pred, current_embeddings = self.model(x, edge_index, edge_label_index, previous_embeddings, num_current_edges,
                                              num_previous_edges)
        return pred, current_embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer

    def _shared_step(self, batch):
        pred, _ = self.forward(batch)
        loss = self.loss_fn(pred, batch.edge_label.type_as(pred))
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric(pred_cont, batch.edge_label.int())
        return loss, avg_pr

    def training_step(self, batch, batch_idx):
        loss, avg_pr = self._shared_step(batch)
        optimizer = self.optimizers()  # Get the optimizer
        self.manual_backward(loss, retain_graph=True)  # Manually handle backward pass
        optimizer.step()  # Update the model parameters
        optimizer.zero_grad()  # Zero the gradients for the next step
        self.log("train_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, avg_pr = self._shared_step(batch)
        self.log("val_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, avg_pr = self._shared_step(batch)
        self.log("test_avg_pr", avg_pr)
        self.log("test_loss", loss)
