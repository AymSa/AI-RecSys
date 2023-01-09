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
from data import BiGraphDataModule
from model import LightGCN


if __name__ == "__main__":
    # mettre dans le fichier de config
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = "../datasets/ml-latest-small/ratings.csv"
    src_name = "userId"
    dst_name = "movieId"
    edge_name = "rating"
    edge_threshold = 3.5
    drop_name = ["timestamp"]

    datamodule = BiGraphDataModule(
        path, drop_name, src_name, edge_name, dst_name, edge_threshold, 1024
    )

    train_adj, R_train, R_val, R_test = datamodule.generate_matrix()

    n_src = datamodule.n_src
    n_dst = datamodule.n_dst

    model = LightGCN(train_adj,R_train.to("cuda"), R_val.to("cuda"), R_test.to("cuda"), n_src, n_dst)

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=10,
        accelerator="gpu",  # TODO
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(model, datamodule)


    test_results = trainer.test(model, dataloaders=datamodule.test_dataloader())


    