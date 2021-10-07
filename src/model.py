import torch
from torch import nn, optim

from torch.nn import functional as F
from torchvision.ops import focal_loss
import pytorch_lightning as pl
from sklearn import metrics

from .bgnn import BGNNA, BGCNA
from .model_help import BaseModel
from .dataset import PairGraphData
from . import MODEL_REGISTRY

class NeighborEmbedding(nn.Module):
    def __init__(self, num_embeddings, out_channels=128, dropout=0.5, cached=True, bias=True, lamda=0.8, share=True):
        super(NeighborEmbedding, self).__init__()

        self.shutcut = nn.Linear(in_features=num_embeddings, out_features=out_channels)

        self.bgnn = BGCNA(in_channels=num_embeddings, out_channels=out_channels,
                          cached=cached, bias=bias, lamda=lamda, share=share)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = out_channels

    def forward(self, x, edge, embedding):
        if not hasattr(self, "edge_index"):
            edge_index = torch.sparse_coo_tensor(*edge)
            self.register_buffer("edge_index", edge_index)
        edge_index = self.edge_index

        embedding = self.bgnn(embedding, edge_index=edge_index)

        embedding = self.dropout(embedding)
        x = F.embedding(x, embedding)
        x = F.normalize(x)

        return x


class InteractionEmbedding(nn.Module):
    def __init__(self, n_drug, n_disease, embedding_dim, dropout=0.5):
        super(InteractionEmbedding, self).__init__()
        self.drug_project = nn.Linear(n_drug, embedding_dim, bias=False)
        self.disease_project = nn.Linear(n_disease, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = embedding_dim

    def forward(self, association_pairs, drug_embedding, disease_embedding):
        drug_embedding = torch.diag(torch.ones(drug_embedding.shape[0], device=drug_embedding.device))
        disease_embedding = torch.diag(torch.ones(disease_embedding.shape[0], device=disease_embedding.device))

        drug_embedding = self.drug_project(drug_embedding)
        disease_embedding = self.disease_project(disease_embedding)

        drug_embedding = F.embedding(association_pairs[0,:], drug_embedding)
        disease_embedding = F.embedding(association_pairs[1,:], disease_embedding)

        associations = drug_embedding*disease_embedding

        associations = F.normalize(associations)
        associations = self.dropout(associations)
        return associations

class InteractionDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dims=(256, 64), out_channels=1, dropout=0.5):
        super(InteractionDecoder, self).__init__()
        decoder = []
        in_dims = [in_channels]+list(hidden_dims)
        out_dims = hidden_dims
        for in_dim, out_dim in zip(in_dims, out_dims):
            decoder.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            decoder.append(nn.ReLU(inplace=True))
            decoder.append(nn.Dropout(dropout))
        decoder.append(nn.Linear(hidden_dims[-1], out_channels))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(x)


@MODEL_REGISTRY.register()
class WBNCF(BaseModel):
    DATASET_TYPE = "PairGraphDataset"
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WBNCF model config")
        parser.add_argument("--embedding_dim", default=64, type=int, help="编码器关联嵌入特征维度")
        parser.add_argument("--neighbor_embedding_dim", default=32, type=int, help="编码器邻居特征维度")
        parser.add_argument("--hidden_dims", type=int, default=(64, 32), nargs="+", help="解码器每层隐藏单元数")
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--pos_weight", type=float, default=1.0, help="no used, overwrited, use for bce loss")
        parser.add_argument("--alpha", type=float, default=0.5, help="use for focal loss")
        parser.add_argument("--gamma", type=float, default=2.0, help="use for focal loss")
        parser.add_argument("--lamda", type=float, default=0.8, help="weight for bgnn")
        parser.add_argument("--loss_fn", type=str, default="focal", choices=["bce", "focal"])
        parser.add_argument("--separate", default=False, action="store_true")
        return parent_parser

    def __init__(self, n_drug, n_disease, embedding_dim=64, neighbor_embedding_dim=32, hidden_dims=(64, 32),
                 lr=5e-4, dropout=0.5, pos_weight=1.0, alpha=0.5, gamma=2.0, lamda=0.8,
                 loss_fn="focal", separate=False, **config):
        super(WBNCF, self).__init__()
        # lr=0.1
        self.n_drug = n_drug
        self.n_disease = n_disease
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims

        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("gamma", torch.tensor(gamma))
        "rank bce mse focal"
        self.loss_fn_name = loss_fn
        share = not separate
        self.drug_neighbor_encoder = NeighborEmbedding(num_embeddings=n_drug,
                                                       out_channels=neighbor_embedding_dim,
                                                       dropout=dropout, lamda=lamda, share=share)
        self.disease_neighbor_encoder = NeighborEmbedding(num_embeddings=n_disease,
                                                          out_channels=neighbor_embedding_dim,
                                                          dropout=dropout, lamda=lamda, share=share)
        self.interaction_encoder = InteractionEmbedding(n_drug=n_drug, n_disease=n_disease,
                                                        embedding_dim=embedding_dim, dropout=dropout)
        merged_dim = self.disease_neighbor_encoder.output_dim\
                     +self.drug_neighbor_encoder.output_dim\
                     +self.interaction_encoder.output_dim
        self.decoder = InteractionDecoder(in_channels=merged_dim, hidden_dims=hidden_dims, dropout=dropout,
                                          )
        self.config = config
        self.lr = lr
        self.save_hyperparameters()


    def forward(self, interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding):

        drug_neighbor_embedding = self.drug_neighbor_encoder(interaction_pairs[0,:], drug_edge, drug_embedding)
        disease_neighbor_embedding = self.disease_neighbor_encoder(interaction_pairs[1,:], disease_edge, disease_embedding)
        interaction_embedding = self.interaction_encoder(interaction_pairs, drug_embedding, disease_embedding)

        embedding = torch.cat([drug_neighbor_embedding, interaction_embedding, disease_neighbor_embedding], dim=-1)
        score = self.decoder(embedding)
        return score.reshape(-1)


    def loss_fn(self, predict, label, u, v, u_edge, v_edge, reduction="sum"):
        bce_loss = self.bce_loss_fn(predict, label, self.pos_weight)
        focal_loss = self.focal_loss_fn(predict, label, gamma=self.gamma, alpha=self.alpha)
        mse_loss = self.mse_loss_fn(predict, label, self.pos_weight)
        rank_loss = self.rank_loss_fn(predict, label)

        u_graph_loss = self.graph_loss_fn(x=u, edge=u_edge, cache_name="ul",
                                          # topk=5,
                                          topk = self.config["drug_neighbor_num"],
                                          reduction=reduction)
        v_graph_loss = self.graph_loss_fn(x=v, edge=v_edge, cache_name="vl",
                                          # topk=5,
                                          topk = self.config["disease_neighbor_num"],
                                          reduction=reduction)
        graph_loss = u_graph_loss * self.lambda1 + v_graph_loss * self.lambda2


        loss = {}
        loss.update(bce_loss)
        loss.update(focal_loss)
        loss.update(mse_loss)
        loss.update(rank_loss)
        loss["loss_graph"] = graph_loss
        loss["loss_graph_u"] = u_graph_loss
        loss["loss_graph_v"] = v_graph_loss
        loss["loss"] = loss[f"loss_{self.loss_fn_name}"]+graph_loss

        return loss


    def step(self, batch:PairGraphData):
        interaction_pairs = batch.interaction_pair
        label = batch.label
        drug_edge = batch.u_edge
        disease_edge = batch.v_edge
        drug_embedding = batch.u_embedding
        disease_embedding = batch.v_embedding
        u = self.interaction_encoder.drug_project.weight.T
        v = self.interaction_encoder.disease_project.weight.T

        predict = self.forward(interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding)
        if not self.training:
            predict = predict[batch.valid_mask.reshape(*predict.shape)]
            label = label[batch.valid_mask]
        ans = self.loss_fn(predict=predict, label=label, u=u, v=v, u_edge=drug_edge, v_edge=disease_edge)
        ans["predict"] = predict
        ans["label"] = label
        return ans


    def training_step(self, batch, batch_idx=None):
        return self.step(batch)

    def validation_step(self, batch, batch_idx=None):
        return self.step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(lr=self.lr, params=self.parameters(), weight_decay=1e-4)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.05*self.lr, max_lr=self.lr,
                                                   gamma=0.95, mode="exp_range", step_size_up=4,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]

    @property
    def lambda1(self):
        max_value = 0.125
        value = self.current_epoch/18.0*max_value
        return torch.tensor(value, device=self.device)

    @property
    def lambda2(self):
        max_value = 0.0625
        value = self.current_epoch / 18.0 * max_value
        return torch.tensor(value, device=self.device)



