import os
from datetime import datetime
import numpy as np
import copy
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from models.models import EdgeRolandGNN
from models.lightning_modules import LightningGNN

DATASET_PATH = os.environ.get("PATH_DATASET", "data/")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "experiments")
L.seed_everything(1234)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
root_dir = os.path.join(CHECKPOINT_PATH, "edge_level")
# os.makedirs(root_dir, exist_ok=True)
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # Loading data
    data = BitcoinOTC(root=f"{DATASET_PATH}/BitcoinOTC")
    data.name = "Bitcoin"
    hidden_conv1, hidden_conv2 = 64, 32
    for data_index in range(len(data)-1):
        print(f"Train on timestamp:{data_index}, test:{data_index+1}")
        # Preparing train and validation data
        if data_index == 0:
            num_previous_edges = 0
            num_nodes = data[data_index].num_nodes
            previous_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(num_nodes)]),
                                   torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(num_nodes)])]
        else:
            num_previous_edges = data[data_index - 1].num_edges
            _, previous_embeddings = lightningModule.forward(train_data)
        snapshot = data[data_index]
        num_nodes = snapshot.num_nodes
        num_current_edges = snapshot.num_edges
        snapshot.x = torch.Tensor([[1] for _ in range(snapshot.num_nodes)])
        transform = RandomLinkSplit(num_val=0.0, num_test=0.25)
        train_data, _, val_data = transform(snapshot)
        train_data.previous_embeddings = previous_embeddings
        train_data.num_current_edges = snapshot.num_edges
        train_data.num_previous_edges = num_previous_edges
        val_data.previous_embeddings = previous_embeddings
        val_data.num_current_edges = snapshot.num_edges
        val_data.num_previous_edges = num_previous_edges
        ## Preparing test data
        test_data = copy.deepcopy(data[data_index+1])
        test_data.num_current_edges = test_data.num_edges
        test_data.num = test_data.num_nodes
        test_data.x = torch.Tensor([[1] for _ in range(test_data.num_nodes)])
        future_neg_edge_index = negative_sampling(
            edge_index=test_data.edge_index,  # positive
            num_nodes=test_data.num_nodes,
            num_neg_samples=test_data.num_edges
        )
        num_pos_edges = test_data.num_edges
        test_data.edge_label = torch.Tensor(
            np.array([1 for _ in range(num_pos_edges)] + [0 for _ in range(num_pos_edges)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)
        test_data.previous_embeddings = previous_embeddings
        test_data.num_previous_edges = train_data.num_current_edges
        print(train_data)
        print(val_data)
        print(test_data)
        # Start training and testing.
        train_loader = DataLoader([train_data], batch_size=1)
        val_loader = DataLoader([val_data], batch_size=1)
        test_loader = DataLoader([test_data], batch_size=1)
        # Defining the model
        model = EdgeRolandGNN(1, hidden_conv1, hidden_conv2, data[0].num_nodes, 0.0, "GCN", "gru")
        model.reset_parameters()
        print(model)
        lightningModule = LightningGNN(model)
        experiments_dir = f"{root_dir}/{data.name}/{experiment_datetime}/timestamp_{data_index}"
        csv_logger = CSVLogger(experiments_dir,version="")
        trainer = L.Trainer(default_root_dir=experiments_dir,
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_avg_pr")],
                            accelerator="auto",
                            devices="auto",
                            enable_progress_bar=True,
                            logger=csv_logger,
                            max_epochs=100
                            )
        trainer.fit(lightningModule, train_loader, val_loader)
        trainer.test(lightningModule,test_loader)