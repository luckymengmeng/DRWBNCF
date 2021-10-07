import os
import torch
from torch.utils import data
import pandas as pd
import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold

def load_DRIMC(root_dir="dataset/drimc", name="c", reduce=True):
    """ C drug:658, disease:409 association:2520 (False 2353)
        PREDICT drug:593, disease:313 association:1933 (Fdataset)
        LRSSL drug: 763, disease:681, association:3051
    """
    drug_chemical = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_chemical.txt"), sep="\t", index_col=0)
    drug_domain = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_domain.txt"), sep="\t", index_col=0)
    drug_go = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_go.txt"), sep="\t", index_col=0)
    disease_sim = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dg.txt"), sep="\t", index_col=0)
    if reduce:
        drug_sim =  (drug_chemical+drug_domain+drug_go)/3
    else:
        drug_sim = drug_chemical
    drug_disease = pd.read_csv(os.path.join(root_dir, f"{name}_admat_dgc.txt"), sep="\t", index_col=0).T
    if name=="lrssl":
        drug_disease = drug_disease.T
    rr = drug_sim.to_numpy(dtype=np.float32)
    rd = drug_disease.to_numpy(dtype=np.float32)
    dd = disease_sim.to_numpy(dtype=np.float32)
    rname = drug_sim.columns.to_numpy()
    dname = disease_sim.columns.to_numpy()
    return {"drug":rr,
            "disease":dd,
            "Wrname":rname,
            "Wdname":dname,
            "didr":rd.T}

def load_HDVD(root_dir="dataset/hdvd"):
    """drug:219, virus:34, association: 455"""
    dd = pd.read_csv(os.path.join(root_dir, "virussim.csv"), index_col=0).to_numpy(np.float32)
    rd = pd.read_csv(os.path.join(root_dir, "virusdrug.csv"), index_col=0)
    rr = pd.read_csv(os.path.join(root_dir, "drugsim.csv"), index_col=0).to_numpy(np.float32)
    rname = rd.index.to_numpy()
    dname = rd.columns.to_numpy()
    rd = rd.to_numpy(np.float32)
    return {"drug":rr,
            "disease":dd,
            "Wrname":rname,
            "Wdname":dname,
            "didr":rd.T}

class Dataset(data.Dataset):
    def __init__(self, name="Cdataset", mode="train", data=None,
                 train_edge=None, train_label=None,
                 test_edge=None, test_label=None,
                 drug_neighbor=None, disease_neighbr=None, num_neighbor=3):
        super(Dataset, self).__init__()
        assert name in ["Cdataset", "Fdataset", "DNdataset", "lrssl", "hdvd"]
        if data is None:
            if name=="lrssl":
                old_data = load_DRIMC(name=name)
            elif name=="hdvd":
                old_data = load_HDVD()
            else:
                old_data = scio.loadmat(f"dataset/{name}.mat")
            data = {}
            data["drug"] = old_data["drug"].astype(np.float)
            data["disease"] = old_data["disease"].astype(np.float)
            data["Wrname"] = old_data["Wrname"].reshape(-1)
            data["Wdname"] = old_data["Wdname"].reshape(-1)
            data["didr"] = old_data["didr"].T
            print(f"dataset:{name}, drug:{data['drug'].shape[0]}, disease:{data['disease'].shape[0]}")
        self.data = data
        self.name = name
        self.mode = mode
        self.drug_sim = data["drug"]
        self.disease_sim = data["disease"]
        self.drug_name = data["Wrname"]
        self.disease_name = data["Wdname"]
        self.interaction = data["didr"]

        self.num_durg = self.drug_sim.shape[0]
        self.num_disease = self.disease_sim.shape[0]

        self.train_edge = train_edge
        self.train_label= train_label
        self.test_edge = test_edge
        self.test_label = test_label

        if drug_neighbor is None or disease_neighbr is None:
            drug_neighbor, disease_neighbr = self.build_sim_neighbor(num_neighbor=num_neighbor)
        self.num_neighbor = num_neighbor
        self.drug_neighbor = drug_neighbor
        self.disease_neighbor = disease_neighbr

        assert self.drug_sim.shape[0] == self.drug_sim.shape[1]
        assert self.disease_sim.shape[0] == self.disease_sim.shape[1]
        assert self.interaction.shape[0] == self.drug_sim.shape[0]
        assert self.interaction.shape[1] == self.disease_sim.shape[0]

        self.edge = self.train_edge if mode=="train" else self.test_edge
        self.label = self.train_label if mode=="train" else self.test_label
        if self.label is not None:
            pos_num = self.label.sum()
            neg_num = len(self.label)-pos_num
            self.pos_weight = neg_num/pos_num

    def split(self, n_splits, split_zero=True, seed=666):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        if n_splits==1:
            col, row = np.meshgrid(np.arange(self.interaction.shape[1]),
                                   np.arange(self.interaction.shape[0]))
            train_edge = np.stack([row.reshape(-1), col.reshape(-1)])
            train_label = self.interaction.reshape(-1).astype(np.float32)
            yield Dataset(name=self.name, data=self.data,
                          train_edge=train_edge, train_label=train_label,
                          test_edge=train_edge, test_label=train_label,
                          drug_neighbor=self.drug_neighbor, disease_neighbr=self.disease_neighbor,
                          num_neighbor=self.num_neighbor)
            return

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pos_row, pos_col = np.nonzero(self.interaction)
        neg_row, neg_col = np.nonzero(1-self.interaction)
        assert len(pos_row)+len(neg_row)==np.prod(self.interaction.shape)
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row), kfold.split(neg_row)):
            if not split_zero:
                test_neg_idx = np.arange(len(neg_row))
            train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
            train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
            test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
            test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
            train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
            train_label = np.concatenate([np.ones(train_pos_edge.shape[1]),
                                          np.zeros(train_neg_edge.shape[1])]).astype(np.float32)
            test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
            test_label = np.concatenate([np.ones(test_pos_edge.shape[1]),
                                         np.zeros(test_neg_edge.shape[1])]).astype(np.float32)

            train_idx = np.arange(train_label.shape[0])
            test_idx = np.arange(test_label.shape[0])
            train_idx = np.random.permutation(train_idx)
            test_idx = np.random.permutation(test_idx)
            train_edge = train_edge[:,train_idx]
            train_label = train_label[train_idx]
            test_edge = test_edge[:,test_idx]
            test_label = test_label[test_idx]

            yield Dataset(name=self.name, data=self.data,
                          train_edge=train_edge, train_label=train_label,
                          test_edge=test_edge, test_label=test_label,
                          drug_neighbor=self.drug_neighbor, disease_neighbr=self.disease_neighbor,
                          num_neighbor=self.num_neighbor)

    def split_train_test(self):
        return (Dataset(name=self.name, mode="train", data=self.data, train_edge=self.train_edge, train_label=self.train_label, num_neighbor=self.num_neighbor),
                Dataset(name=self.name, mode="test", data=self.data, test_edge=self.test_edge, test_label=self.test_label, num_neighbor=self.num_neighbor))

    def col_local_leave_one_split(self, col_idx):
        mask = np.ones_like(self.interaction, dtype="bool")
        col, row = np.meshgrid(np.arange(mask.shape[1]), mask.shape[0])
        mask = mask[:,col_idx] = False
        train_edge = np.stack([row[mask], col[mask]])
        train_label = self.interaction[mask].astype(np.float32)
        test_edge = np.stack([np.arange(mask.shape[0]), np.ones(mask.shape[0], dtype="int")*col_idx])
        test_label = self.interaction[:, col_idx].astype(np.float32)
        yield Dataset(name=self.name, data=self.data,
                      train_edge=train_edge, train_label=train_label,
                      test_edge=test_edge, test_label=test_label,
                      drug_neighbor=self.drug_neighbor, disease_neighbr=self.disease_neighbor,
                      num_neighbor=self.num_neighbor)

    def col_split(self):
        for col in range(self.interaction.shape[1]):
            yield from self.col_local_leave_one_split(col)

    def build_cos_neighbor(self, num_neighbor):
        interaction = np.zeros_like(self.interaction)
        interaction[self.train_edge[0], self.train_edge[1]] = self.train_label
        drug_neighbor = self.get_cos_neighbor(interaction, num_neighbor=num_neighbor)
        disease_neighbor = self.get_cos_neighbor(interaction.T, num_neighbor=num_neighbor)
        return drug_neighbor, disease_neighbor

    def get_cos_neighbor(self, feature, num_neighbor):
        feature = feature/np.linalg.norm(feature, dim=1, keepdims=True)
        x = feature[:,np.newaxis,:]
        y = feature[np.newaxis,:,:]
        score = np.sum(x*y, axis=-1)
        neighbor_idx = np.argpartition(score, kth=num_neighbor, axis=1)[:,:num_neighbor]
        return neighbor_idx

    def build_sim_neighbor(self, num_neighbor):
        drug_neighbor = np.argpartition(self.drug_sim, kth=num_neighbor, axis=1)[:,:num_neighbor]
        disease_neighbor = np.argpartition(self.disease_sim, kth=num_neighbor, axis=1)[:,:num_neighbor]
        return drug_neighbor, disease_neighbor

    def __len__(self):
        return len(self.label)
        # return 500

    def __getitem__(self, index):
        drug, disease = self.edge[:, index]
        drug_neighbor = self.drug_neighbor[drug]
        disease_neighbor = self.disease_neighbor[disease]
        label = self.label[index]
        return drug, disease, label, drug_neighbor, disease_neighbor

    def export(self, save_dir):
        train_file = os.path.join(save_dir, "train.csv")
        dev_file = os.path.join(save_dir, "dev.csv")
        if self.train_edge is not None:
            pos_idx = np.nonzero(self.train_label)[0]
            user_id = self.train_edge[0,pos_idx]+1
            item_id = self.train_edge[1,pos_idx]+1
            train_data = {"user_id":user_id,
                          "item_id":item_id,
                          }
            train_data = pd.DataFrame(train_data)
            train_data["time"] = 0
            train_data.to_csv(train_file, index=False)
        if self.test_edge is not None:
            pos_idx = np.nonzero(self.test_label)[0]
            user_id = self.test_edge[0, pos_idx] + 1
            item_id = self.test_edge[1, pos_idx] + 1
            test_data = {"user_id":user_id,
                         "item_id":item_id,
                         "neg_items":[0]}
            test_data = pd.DataFrame(test_data)
            test_data["time"] = 0
            test_data.to_csv(dev_file, index=False)

