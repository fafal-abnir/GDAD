import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip


class DGraphFin(InMemoryDataset):
    r"""The DGraphFin networks from the
        `"DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection"
        <https://arxiv.org/abs/2207.03579>`_ paper.
        It is a directed, unweighted dynamic graph consisting of millions of
        nodes and edges, representing a realistic user-to-user social network
        in financial industry.
        Node represents a Finvolution user, and an edge from one
        user to another means that the user regards the other user
        as the emergency contact person. Each edge is associated with a
        timestamp ranging from 1 to 821 and a type of emergency contact
        ranging from 0 to 11.


        Args:
            root (str): Root directory where the dataset should be saved.
            edge_window_size (int, optional): The window size for grouping edges. (default: :obj:'7' weekly)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            force_reload (bool, optional): Whether to re-process the dataset.
                (default: :obj:`False`)

        **STATS:**

        .. list-table::
            :widths: 10 10 10 10
            :header-rows: 1

            * - #nodes
              - #edges
              - #features
              - #classes
            * - 3,700,550
              - 4,300,999
              - 17
              - 2
        """

    url = "https://dgraph.xinye.com"

    def __init__(self,
                 root: str,
                 edge_window_size: int = 7,
                 num_windows: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False
                 ) -> None:
        self.edge_window_size = edge_window_size
        self.num_windows = num_windows
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self) -> str:
        return 'DGraphFin.zip'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_node_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        return 2

    def process(self) -> None:
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        path = osp.join(self.raw_dir, "dgraphfin.npz")

        with np.load(path) as loader:
            x = loader['x']
            y = loader['y']
            edge_index = loader['edge_index']
            edge_timestamp = loader['edge_timestamp']
            edge_type = loader['edge_type']
            max_timestamps = np.max(edge_timestamp)
            # Filter with respect timestamp.
            edge_mask = edge_timestamp < self.edge_window_size * self.num_windows
            adjusted_edge_timestamp = edge_timestamp[edge_mask]
            adjusted_edge_index = edge_index[edge_mask]
            adjusted_edge_type = edge_type[edge_mask]
            relevant_nodes = set()
            relevant_nodes.update(adjusted_edge_index.flatten().tolist())
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(relevant_nodes))}
            adjusted_x = torch.tensor(x[list(relevant_nodes)], dtype=torch.float)
            adjusted_edge_index = torch.tensor(
                [[node_mapping[edge[0]] for edge in adjusted_edge_index],
                 [node_mapping[edge[1]] for edge in adjusted_edge_index]],
                dtype=torch.long
            )
            data_list = []
            for timestamp in range(1, max_timestamps, self.edge_window_size):
                edge_mask = (timestamp <= adjusted_edge_timestamp) & (
                            adjusted_edge_timestamp < timestamp + self.edge_window_size)
                filtered_edge_index = adjusted_edge_index[:, edge_mask]
                filtered_edge_type = adjusted_edge_type[edge_mask]
                data = Data(
                    x=adjusted_x,
                    edge_index=torch.tensor(filtered_edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(filtered_edge_type, dtype=torch.long),
                    y=torch.tensor(y, dtype=torch.long)
                )
                data_list.append(data)
                if len(data_list) >= self.num_windows:
                    break
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        self.save(data_list, self.processed_paths[0])
