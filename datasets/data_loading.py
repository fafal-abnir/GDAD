import torch
from torch_geometric.datasets import BitcoinOTC
from datasets.dgraph import DGraphFin


def get_dataset(name: str, root_dir):
    if name == "BitcoinOTC":
        dataset = BitcoinOTC(root=f"{root_dir}/BitcoinOTC")
        dataset.x = torch.Tensor([[1] for _ in range(dataset.num_nodes)])
        dataset.name = "BitcoinOTC"
    elif name == "DGraphFin":
        dataset = DGraphFin(root=f"{root_dir}/DGraphFin", edge_window_size=7, num_windows=10)
        dataset.name = "DGraphFin"
    else:
        raise Exception("Unknown dataset.")
    return dataset
