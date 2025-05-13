import os
import os.path as osp
import shutil
import torch
from typing import Callable, Optional
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
from torch_geometric.graphgym.config import cfg


class OCBDataset(InMemoryDataset):
    r"""The OCB dataset (Dong et al., 2023) is pioneering benchmark in the circuit domain, specifically designed for optimizing both analog circuit topologies and device parameters. Ckt-Bench101 comprises 10,000 operational amplifier (OpAmp) circuits, each topology represented as a directed acyclic graph(DAG). Ckt-Bench301 includes 47,248 OpAmp circuits, after excluding 2,752 invalid simulation results from the original 50,000 entries. For regression tasks, performance metrics for these circuits have been meticulously extracted using a circuit simulator. The OCB dataset provides critical performance metrics such as gain, bandwidth, phase margin, and a figure of merit (a composite metric of these parameters) as labels.
    
    Size of Dataset: Ckt-Bench101 (10,000 circuits), Ckt-Bench301 (47,248 circuits).
    Split: Train/Val/Test (8/1/1)
    Number of node features: 2
    Number of edge features: 3
    Performance Metric: MAE

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    
    names = ["ckt_bench_101", "ckt_bench_301"]
    root_directory = "OCB" # Root directory where the data located
    files = {
        "ckt_bench_101": f"{root_directory}/CktBench101/ckt_bench_101.pt",
        "ckt_bench_301": f"{root_directory}/CktBench301/ckt_bench_301.pt",
    }

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name
        assert self.name in self.names
        
        super().__init__(root, transform, pre_transform, pre_filter)

        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        else:
            raise ValueError(f"Split '{split}' found, but expected either " f"'train', 'val', or 'test'")
        
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        name = self.files[self.name].split("/")[-1][:-3]
        return [f"{name}.pt"]
    
    @property
    def processed_file_names(self):
        return ["train_data.pt", "val_data.pt", "test_data.pt"]

    def download(self):
        dst_path = self.raw_dir
        os.makedirs(dst_path, exist_ok=True)
        src_file = self.files[self.name]
        shutil.copy(src_file, dst_path)
        self.data_split()

    def data_split(self):
        inputs = torch.load(self.raw_paths[0])
        split_ratios = cfg.dataset.split
        train, temp = train_test_split(inputs, test_size=1 - split_ratios[0], random_state=42)
        val, test = train_test_split(
            temp, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]), random_state=42
        )
        torch.save([train, val, test], self.raw_paths[0])
    
    def process(self):
        inputs = torch.load(self.raw_paths[0])
        for i in range(len(inputs)):
            data_list = [Data(**data_dict) for data_dict in inputs[i]]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[i])

    def __repr__(self) -> str:
        return f"{self.name}({len(self)})"
    

if __name__ == "__main__":
    dataset = OCBDataset(root="./datasets", name="ckt_bench_101")
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
