import pandas as pd
import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything
import numpy as np
import pandas as pd
import mlflow
from torch.utils.data import Dataset, DataLoader
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count
from torch_geometric.utils import structured_negative_sampling
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import stratify_split
import scipy.sparse as sp 

import random 

class TrainBiGraphDataset(Dataset):
    def __init__(self, R, n_src, n_dst, dst_name) -> None:
        super().__init__()

        self.R = R
        self.n_src = n_src
        self.n_dst = n_dst 
        self.dst_name = dst_name
        self.src_dst_pos = list(R.keys())



    def __getitem__(self, index):
        """
        Retourne le couple (i,j,k) avec i indices de la src, j la connexion positive de la dst et k une connexion neg
        qu'on utilisera pour calculer la loss
        
        """
        
        def sample_neg(x):
            while True:
                neg_id = random.randint(0, self.n_dst - 1)
                if neg_id not in x:
                    return neg_id

        src, dst_pos = self.src_dst_pos[index]
        
        _, interact = self.R[src].nonzero()

        dst_neg = sample_neg(interact)

        return src, dst_pos, dst_neg

    def __len__(self):
        return len(self.src_dst_pos)


class EvalBiGraphDataset(Dataset):

    def __init__(self, data, src_name) -> None:
        super().__init__()
        self.data = data 
        self.src_name = src_name
        self.src_ids = self.data[self.src_name].unique()

    def __getitem__(self, index):
        return self.src_ids[index]

    def __len__(self):
        return len(self.src_ids)

class BiGraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_path,
        timestamp_name,
        src_name,
        edge_name,
        dst_name,
        edge_threshold,
        batch_size,
        train_val_test : list[float, float, float] = [0.8,0.1,0.1]
    ) -> None:
        super().__init__()

        assert sum(train_val_test) == 1.

        self.df_path = df_path
        self.timestamp_name = timestamp_name
        self.src_name = src_name
        self.edge_name = edge_name
        self.dst_name = dst_name
        self.edge_threshold = edge_threshold
        self.batch_size = batch_size
        self.train_val_test = train_val_test
        self.num_workers = cpu_count()
        self.train_df, self.val_df, self.test_df = self.setup_data()
        self.interact_status_train, self.R_train = self._init_interaction(self.train_df) 
        self.interact_status_val, self.R_val = self._init_interaction(self.val_df)
        self.interact_status_test, self.R_test = self._init_interaction(self.test_df)

    def pre_process(self, data):
        #Etape de pre processing deja 

        data = data.drop_duplicates() #Supprime les duplicates 
        
        #On doit aussi retravailler sur les indexes 

        self.n_src = len(data[self.src_name].unique())
        self.n_dst = len(data[self.dst_name].unique())

        self.label_src = LabelEncoder()
        self.label_dst = LabelEncoder()

        data[self.src_name] = self.label_src.fit_transform(data[self.src_name])
        data[self.dst_name] = self.label_dst.fit_transform(data[self.dst_name])

        ###Faire le threshold ici !
        data = data[data[self.edge_name] >= self.edge_threshold]
        return data
    
    def setup_data(self):
        
        df = pd.read_csv(self.df_path)
        self.df = self.pre_process(df)

        return stratify_split(self.df, self.train_val_test, self.src_name)

    def _init_interaction(self, data):

        interact_status = (
            data.groupby(self.src_name)[self.dst_name]
            .apply(set)
            .reset_index()
            .rename(columns={self.dst_name: self.dst_name + "_interacted"})
        )
        R = sp.dok_matrix((self.n_src, self.n_dst), dtype=np.float32)
        R[data[self.src_name], data[self.dst_name]] = 1.0

        return interact_status, R

    def generate_matrix(self):

        adj_mat = sp.dok_matrix(
            (self.n_src+ self.n_dst, self.n_src+ self.n_dst), dtype=np.float32
        )

        adj_mat = adj_mat.tolil()
        R = self.R_train.tolil()

        adj_mat[: self.n_src, self.n_src :] = R
        adj_mat[self.n_src :, : self.n_src] = R.T
        adj_mat = adj_mat.todok()
        print("Create adjacency matrix.")
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(d_mat_inv)
        print("Normalize adjacency matrix.")
        
        
        return self.sp_to_tensor(norm_adj_mat), self.sp_to_tensor(self.R_train), self.sp_to_tensor(self.R_val), self.sp_to_tensor(self.R_test)

    def sp_to_tensor(self, sp_mat):

        sp_mat = sp_mat.tocoo()
        return torch.sparse.FloatTensor(torch.LongTensor([sp_mat.row.tolist(), sp_mat.col.tolist()]),
                              torch.Tensor(sp_mat.data), size = sp_mat.shape) #Float Sparse Tensor !!! (Surement incompatible avec torch geo)


    def train_dataloader(self):
        #interact status est propre a train_data 
        train_dataset = TrainBiGraphDataset(self.R_train, self.n_src, self.n_dst, self.dst_name)

        return DataLoader(train_dataset, 
            batch_size=self.batch_size, 
            shuffle= True, 
            pin_memory=True, 
            num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataset = EvalBiGraphDataset(self.test_df, self.src_name)
        return DataLoader(test_dataset, 
            batch_size=self.batch_size, 
            shuffle= False, 
            pin_memory=True, 
            num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = EvalBiGraphDataset(self.val_df, self.src_name)
        return DataLoader(val_dataset, 
            batch_size=self.batch_size, 
            shuffle= False, 
            pin_memory=True, 
            num_workers=self.num_workers)

if __name__ == '__main__':
    

    path = "../datasets/ml-latest-small/ratings.csv"
    src_name = "userId"
    dst_name = "movieId"
    edge_name = "rating"
    edge_threshold = 3.5
    drop_name = ["timestamp"]


    datamodule = BiGraphDataModule(path, drop_name, src_name, edge_name, dst_name, edge_threshold, 128) 
    adj, r_train, r_val, r_test = datamodule.generate_matrix()

    print(adj)
    idx = torch.arange(10)
    