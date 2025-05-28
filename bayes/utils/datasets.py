import json
import os
import random
from typing import List, Union

import matplotlib.pyplot as plt
import networkx
import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from numpy import genfromtxt
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import utils
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import (
    Amazon,
    CitationFull,
    Planetoid,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.datasets.graph_generator import GridGraph
from torch_geometric.transforms import BaseTransform


def set_manual_seed(seed=1):
    """
    The set_manual_seed function sets the seed for random number generation in Python,
    NumPy and PyTorch.

    :param seed: Set the seed for all of the random number generators used in pytorch
    :return: Nothing
    """
    assert type(isinstance(seed, int))
    os.environ["PYTHONASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.benchmark = True


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for various libraries to ensure reproducibility.

    This function sets the random seed for the NumPy, Python's built-in random module,
    PyTorch CPU and GPU, and other related configurations to ensure that random
    operations produce consistent results across different runs.

    Parameters:
        seed: The seed value to set for random number generation. Default is 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_dataset(dataset_name, dataset_type="Planetoid"):
    dataset_dir = os.path.join(f"../../datasets/{dataset_type}", dataset_name)
    dataset_path = os.path.join(os.getcwd(), dataset_dir)
    total_nodes = 0
    if not os.path.exists(os.path.join(os.getcwd(), f"../../datasets/{dataset_type}")):
        os.makedirs(os.path.join(os.getcwd(), f"../../datasets/{dataset_type}"))

    if dataset_type == "PygNodePropPredDataset":
        assert dataset_name in ["ogbn-arxiv", "ogbn-proteins"]
        dataset= load_pyg_node_prop_pred_dataset(
            name=dataset_name, root=dataset_path
        )

    elif dataset_type == "WikipediaNetwork":
        assert dataset_name in ["Chameleon", "Squirrel"]
        dataset = load_wikipedianetwork_dataset(
            name=dataset_name, root=dataset_path
        )
        
    else:
        assert dataset_type == "Planetoid" and dataset_name in [
            "Cora",
            "CiteSeer",
            "PubMed",
        ]

        dataset = Planetoid(root=dataset_path, name=dataset_name.lower())
        save_path = os.path.join(dataset_path, f"{dataset_name.lower()}.pt")
        torch.save(dataset, save_path)
        """ else:
            dataset = torch.load(
                os.path.join(dataset_path, f"{dataset_name.lower()}.pt")
            )
        """
        total_nodes = dataset[0].num_nodes

        print("============================================")
        print("Node number (totally): ", total_nodes)
        print("============================================")

        train_mask = dataset[0].train_mask
        num_train_nodes = torch.sum(train_mask).item()
        test_mask = dataset[0].test_mask
        num_test_nodes = torch.sum(test_mask).item()
        valid_mask = dataset[0].val_mask
        num_valid_nodes = torch.sum(valid_mask).item()

        print("Number of train nodes after:", num_train_nodes)
        print("Number of test nodes after:", num_test_nodes)
        print("Number of validation nodes after:", num_valid_nodes)

        print(
            f"Portions: train set = {(num_train_nodes / total_nodes) * 100 :.2f}%, "
            f"test set = {(num_test_nodes / total_nodes) * 100 :.2f}%, "
            f"validation set ={(num_valid_nodes / total_nodes) * 100 :.2f}%"
        )
        print("============================================")

    return dataset


def load_pyg_node_prop_pred_dataset(name="ogbn-arxiv", root="."):
    """
    Load the PygNodePropPredDataset dataset and generate train, validation, and test masks based on percentages.

    Args:
        name (str): Name of the dataset (default is "Chameleon").
        root (str): Root directory for dataset storage.
    Returns:
        tuple: A tuple containing the Data object and the number of classes.
    """
    dataset = PygNodePropPredDataset(root=root, name=name, transform=T.TargetIndegree())
    data = dataset[0]
    data.y = data.y.squeeze(dim=1)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    # Initialize masks
    num_nodes = data.num_nodes
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Set masks
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    total_nodes = data.num_nodes

    train_count = data.train_mask.sum().item()
    val_count = data.val_mask.sum().item()
    test_count = data.test_mask.sum().item()
    print("============================================")

    print(f"Train nodes: {train_count} ({100 * train_count / total_nodes:.2f}%)")
    print(f"Validation nodes: {val_count} ({100 * val_count / total_nodes:.2f}%)")
    print(f"Test nodes: {test_count} ({100 * test_count / total_nodes:.2f}%)")

    print("Node number (totally): ", total_nodes)
    print("============================================")

    return dataset



def load_wikipedianetwork_dataset(name="Chameleon", root="."):
    """
    Load the WikipediaNetwork dataset and generate train, validation, and test masks based on percentages.

    Args:
        name (str): Name of the dataset (default is "Chameleon").
        root (str): Root directory for dataset storage.
    Returns:
        tuple: A tuple containing the Data object and the number of classes.
    """
    dataset = WikipediaNetwork(root=root, name=name, transform=T.NormalizeFeatures())
    # Get the data object once and modify it directly
    data = dataset[0].clone()  # Create a copy to work with

    train_mask = data.train_mask
    num_train_nodes = torch.sum(train_mask).item()
    test_mask = data.test_mask
    num_test_nodes = torch.sum(test_mask).item()
    valid_mask = data.val_mask
    num_valid_nodes = torch.sum(valid_mask).item()
    total_nodes = data.num_nodes

    # Modify the masks
    data.train_mask = train_mask[:, 0]
    data.test_mask = test_mask[:, 0]
    data.val_mask = valid_mask[:, 0]  # Note: it's val_mask, not valid_mask

    # Now assign the modified data back to the dataset
    dataset._data = data

    print("============================================")
    print("Node number (totally): ", total_nodes)
    print("============================================")

    print("Number of train nodes after:", torch.sum(data.train_mask).item())
    print("Number of test nodes after:", torch.sum(data.test_mask).item())
    print("Number of validation nodes after:", torch.sum(data.val_mask).item())

    print(
        f"Portions: train set = {(torch.sum(data.train_mask).item() / total_nodes) * 100 :.2f}%, "
        f"test set = {(torch.sum(data.test_mask).item() / total_nodes) * 100 :.2f}%, "
        f"validation set ={(torch.sum(data.val_mask).item() / total_nodes) * 100 :.2f}%"
    )
    print("============================================")

    return dataset




def visualize_data(data, name, colors=None):
    """
    The visualize_data function takes in a dataset and the name of the dataset.
    It then creates a graph from that data, using NetworkX. It then uses
    spring_layout to create an xyz position for each node.
    The function plots these nodes as points on a 3D plot, with edges
    connecting them.

    :param data: Pass the data object to the function
    :param name: Determine the number of nodes in the graph
    :return: A 3d plot of the graph
    """
    plt.style.use("dark_background")
    edge_index = data.edge_index.numpy()
    num_nodes = 2708 if name.lower() == "cora" else data.num_nodes
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.transpose())
    pos = nx.spring_layout(G, dim=3, seed=779)
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if colors is None:
        colors = np.array([np.linalg.norm(pos[v]) for v in sorted(G)])
        ax.scatter(*node_xyz.T, s=25, ec="w", c=colors, cmap="rainbow")
    else:
        ax.scatter(*node_xyz.T, s=25, ec="w", c=colors)

    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    def _format_axes(ax):
        ax.grid(False)
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    ax.set_axis_off()
    plt.title(name)
    _format_axes(ax)
    fig.tight_layout()

    plt.show()


class NormalizeFeaturesColumnwise(BaseTransform):
    """
    Normalizes the specified attributes column-wise in the given Data or
    HeteroData object.
    """

    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        """
        Performs column-wise normalization on the specified attributes in
        the given Data or HeteroData.

        Args:
            data (Union[Data, HeteroData]): Input Data or HeteroData object
            to be normalized.

        Returns:
            Union[Data, HeteroData]: Normalized Data or HeteroData object.
        """
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    mean = value.mean(dim=0, keepdim=True)
                    std = value.std(dim=0, unbiased=False, keepdim=True).clamp_(
                        min=1e-5
                    )
                    value = (value - mean) / std
                    store[key] = value
        return data
