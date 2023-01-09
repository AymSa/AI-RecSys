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
from torch import optim
from metrics import * 
import scipy.sparse as sp 


def get_user_positive_items(edge_index):
    """
    Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges 

    Returns:
        dict: user -> list of positive items for each 
    """
    
    # key: user_id, val: item_id list
    user_pos_items = {}
    
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        
        if user not in user_pos_items:
            user_pos_items[user] = []
        
        user_pos_items[user].append(item)
        
    return user_pos_items

class LightningNet(pl.LightningModule):
    def __init__(
        self,
        optimizer,
        learning_rate: float,
    ) -> None:
        super(LightningNet, self).__init__()

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lambda_loss = 1e-6

    def _bpr_loss(
        self,
        src_embedding,
        init_src_embedding,
        dst_pos_embedding,
        init_dst_pos_embedding,
        dst_neg_embedding,
        init_dst_neg_embedding,
    ):

        reg_loss = self.lambda_loss * (
            init_src_embedding.norm(2).pow(2)
            + init_dst_pos_embedding.norm(2).pow(2)
            + init_dst_neg_embedding.norm(2).pow(2)
        )

        pos_scores = torch.mul(src_embedding, dst_pos_embedding)
        pos_scores = torch.sum(pos_scores, dim=-1)

        neg_scores = torch.mul(src_embedding, dst_neg_embedding)
        neg_scores = torch.sum(neg_scores, dim=-1)

        bpr_loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores))

        return bpr_loss + reg_loss

    def training_step(self, batch):

        indices_src, indices_pos_dst, indices_neg_dst = batch

        src_embedding, init_src_embedding, dst_embedding, init_dst_embedding = self(self.train_adj)

        src_embedding, init_src_embedding = (
            src_embedding[indices_src],
            init_src_embedding[indices_src],
        )
        dst_pos_embedding, init_dst_pos_embedding = (
            dst_embedding[indices_pos_dst],
            init_dst_embedding[indices_pos_dst],
        )
        dst_neg_embedding, init_dst_neg_embedding = (
            dst_embedding[indices_neg_dst],
            init_dst_embedding[indices_neg_dst],
        )

        loss = self._bpr_loss(
            src_embedding,
            init_src_embedding,
            dst_pos_embedding,
            init_dst_pos_embedding,
            dst_neg_embedding,
            init_dst_neg_embedding,
        )

        self.log("train_loss", loss)
        return loss 


    def validation_step( ### A optimiser plein d'etape se font avant l'etape de get batch 
        self, src_ids, batch_idx: int
    ):  

        excluded_row, excluded_col = (self.R_train.to_dense())[src_ids, :].nonzero(as_tuple=True) #osef on passe tout en dense
        
        
        predicted_R = torch.matmul(self.src_embedding.weight, self.dst_embedding.weight.T)
        
        predicted_R = predicted_R[src_ids]
        
        

        #Donne la liste d'interaction predite -> On met a -inf les éléments fournis lors de l'entrainement
        #Car les utilisateurs ont deja regarder ces films la 
        #On donne un resultat tres petit pour ne pas comptabiliser les éléments deja connecté de notre graph


        predicted_R[excluded_row, excluded_col] = - (1<<10) #np.inf

        # get the top k recommended items for each user
        values_top_k_items, idx_top_k_items = torch.topk(predicted_R, k=self.k)


        ###On a les top k recommandations fournies par le modèle
        #On doit mtn verifier si la qualité est bonne en comparant avec la ground truth !!!!!!!

        #list[list] -> items id connecté aux users i x[0][0] correspond a l'user a la pos 0 et l'id a la pos 0 


        ground_truth = list(sp.lil_matrix(((self.R_val.to_dense())[src_ids, :]).cpu().numpy()).rows)
        #ground_truth = [tuple(el) for el in ground_truth]
        prediction = []
        for user in range(len(src_ids)): #Les ids d'users ne sont pas consistant mais fonctionne comme ca !!!! #TODO
            user_true_relevant_item = ground_truth[user]
            label = list(map(lambda x: x in user_true_relevant_item, idx_top_k_items[user]))
            prediction.append(label)

        prediction = torch.Tensor(np.array(prediction).astype('float'))
        
        recall, precision = RecallPrecision_ATk(ground_truth, prediction, self.k)
        ndcg = NDCGatK_r(ground_truth, prediction, self.k)

        self.log('val_recall', recall)
        self.log('val_precision', precision)
        self.log('val_ndcg', ndcg)

        return recall, precision, ndcg

    def test_step(self, src_ids, batch_idx: int):

        excluded_row, excluded_col = (self.R_train.to_dense())[src_ids, :].nonzero(as_tuple=True) #osef on passe tout en dense
        
        
        predicted_R = torch.matmul(self.src_embedding.weight, self.dst_embedding.weight.T)
        
        predicted_R = predicted_R[src_ids]
        
        

        #Donne la liste d'interaction predite -> On met a -inf les éléments fournis lors de l'entrainement
        #Car les utilisateurs ont deja regarder ces films la 
        #On donne un resultat tres petit pour ne pas comptabiliser les éléments deja connecté de notre graph


        predicted_R[excluded_row, excluded_col] = - (1<<10) #np.inf

        # get the top k recommended items for each user
        values_top_k_items, idx_top_k_items = torch.topk(predicted_R, k=self.k)


        ###On a les top k recommandations fournies par le modèle
        #On doit mtn verifier si la qualité est bonne en comparant avec la ground truth !!!!!!!

        #list[list] -> items id connecté aux users i x[0][0] correspond a l'user a la pos 0 et l'id a la pos 0 


        ground_truth = list(sp.lil_matrix(((self.R_test.to_dense())[src_ids, :]).cpu().numpy()).rows)
        #ground_truth = [tuple(el) for el in ground_truth]
        prediction = []
        for user in range(len(src_ids)): #Les ids d'users ne sont pas consistant mais fonctionne comme ca !!!! #TODO
            user_true_relevant_item = ground_truth[user]
            label = list(map(lambda x: x in user_true_relevant_item, idx_top_k_items[user]))
            prediction.append(label)

        prediction = torch.Tensor(np.array(prediction).astype('float'))
        
        recall, precision = RecallPrecision_ATk(ground_truth, prediction, self.k)
        ndcg = NDCGatK_r(ground_truth, prediction, self.k)

        self.log('test_recall', recall)
        self.log('test_precision', precision)
        self.log('test_ndcg', ndcg)

        return recall, precision, ndcg

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=1e-3)


class LightGCN(LightningNet, pyg_nn.MessagePassing):
    def __init__(
        self, train_adj, R_train, R_val, R_test, src_number, dst_number, latent_dim=64, K=3, self_loop=False, k = 20
    ) -> None:
        super(LightGCN, self).__init__("Optimizer", "LR")
        self.src_number = src_number
        self.dst_number = dst_number
        self.latent_dim = latent_dim
        self.K = K
        self.self_loop = self_loop
        self.R_train = R_train
        self.R_val = R_val
        self.R_test = R_test
        self.k = k

        self.src_embedding = nn.Embedding(src_number, latent_dim) #C'est les seuls parametres du modele
        self.dst_embedding = nn.Embedding(dst_number, latent_dim) #C'est les seuls parametres du modele

        nn.init.normal_(self.src_embedding.weight, std=0.1)
        nn.init.normal_(self.dst_embedding.weight, std=0.1)

        self.train_adj = train_adj.to("cuda:0") #TODO 

    def forward(self, adjacence):

        #     _, edge_norm = gcn_norm(
        #     adjacence, add_self_loops=self.self_loop
        # )  # 1/sqrt(Ni*Nu) pour chaque couple deja fait lors de la construction de la matrice A_ = D-AD-
        initial_embedding = torch.cat(
            [self.src_embedding.weight, self.dst_embedding.weight], dim=0
        )
        list_embeddings = [initial_embedding]

        current_embedding = initial_embedding
        
        for _ in range(self.K):
            current_embedding = self.propagate(
                edge_index=adjacence, x=current_embedding
            )  # , norm = edge_norm)
            list_embeddings.append(current_embedding)

        final_embedding = torch.stack(list_embeddings, dim=1).mean(dim=1)

        final_src_embedding, final_dst_embedding = torch.split(
            final_embedding,
            split_size_or_sections=[self.src_number, self.dst_number],
            dim=0,
        )

        return (
            final_src_embedding,
            self.src_embedding.weight,
            final_dst_embedding,
            self.dst_embedding.weight,
        )

    def message(self, x_j):  # , norm):
        return x_j  # * norm.view(-1, 1)


if __name__ == "__main__":

    from data import BiGraphDataModule

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